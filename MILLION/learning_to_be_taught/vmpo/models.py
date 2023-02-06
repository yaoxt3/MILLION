import numpy as np
import torch
import time

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.models.mlp import MlpModel
from learning_to_be_taught.models.transformer_models import DemonstrationTransformerModel, generate_square_subsequent_mask
from rlpyt.models.running_mean_std import RunningMeanStdModel
from learning_to_be_taught.models.gated_transformer import GatedTransformer, SIZES
from rlpyt.utils.collections import namedarraytuple

RnnState = namedarraytuple("RnnState", ["observations", "length"])


class FfModel(torch.nn.Module):
    """
    Model commonly used in Mujoco locomotion agents: an MLP which outputs
    distribution means, separate parameter for learned log_std, and separate
    MLP for state-value estimate.
    """

    def __init__(
            self,
            observation_shape,
            action_size,
            linear_value_output=True,
            layer_norm=False
    ):
        """Instantiate neural net modules according to inputs."""
        super().__init__()
        self._obs_ndim = len(observation_shape)
        input_size = int(np.prod(observation_shape))
        self.action_size = action_size
        self.layer_norm = torch.nn.LayerNorm(input_size) if layer_norm else None
        self.mu_mlp = MlpModel(
            input_size=input_size,
            hidden_sizes=[512, 256, 256],
            output_size=2 * action_size,
        )
        list(self.mu_mlp.parameters())[-1].data = list(self.mu_mlp.parameters())[-1].data / 100
        list(self.mu_mlp.parameters())[-2].data = list(self.mu_mlp.parameters())[-2].data / 100
        self.v = MlpModel(
            input_size=input_size,
            hidden_sizes=[512, 512, 256],
            output_size=1 if linear_value_output else None,
        )

    def forward(self, observation, prev_action, prev_reward, init_rnn_state=None):
        """
        Compute mean, log_std, and value estimate from input state. Infers
        leading dimensions of input: can be [T,B], [B], or []; provides
        returns with same leading dims.  Intermediate feedforward layers
        process as [T*B,H], with T=1,B=1 when not given. Used both in sampler
        and in algorithm (both via the agent).
        """
        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, _ = infer_leading_dims(observation, self._obs_ndim)
        assert not torch.any(torch.isnan(observation)), 'obs elem is nan'

        obs_flat = observation.reshape(T * B, -1)
        if self.layer_norm:
            obs_flat = torch.tanh(self.layer_norm(obs_flat))
        action = self.mu_mlp(obs_flat)
        mu, std = (action[:, :self.action_size], action[:, self.action_size:])
        std = torch.log(1 + torch.exp(std))  # softplus
        v = self.v(obs_flat).squeeze(-1)

        # Restore leading dimensions: [T,B], [B], or [], as input.
        mu, std, v = restore_leading_dims((mu, std, v), lead_dim, T, B)
        fake_rnn_state = torch.zeros(1, B, 1)
        return mu, std, v, fake_rnn_state

    def update_obs_rms(self, observation):
        if self.normalize_observation:
            self.obs_rms.update(observation)


class FfSharedModel(torch.nn.Module):
    """
    Model commonly used in Mujoco locomotion agents: an MLP which outputs
    distribution means, separate parameter for learned log_std, and separate
    MLP for state-value estimate.
    """

    def __init__(
            self,
            observation_shape,
            action_size,
            hidden_sizes=None,  # None for default (see below).
            linear_value_output=True,
            full_covariance=False,
            layer_norm=False,
    ):
        """Instantiate neural net modules according to inputs."""
        super().__init__()
        self._obs_ndim = len(observation_shape)
        input_size = int(np.prod(observation_shape))
        self.full_covariance = full_covariance
        self.action_size = action_size
        self.shared_features_dim = 256
        self.shared_mlp = MlpModel(
            input_size=input_size,
            hidden_sizes=[512, self.shared_features_dim]
        )
        self.mu_head = MlpModel(
            input_size=self.shared_features_dim,
            hidden_sizes=[256],
            output_size=action_size + np.sum(1 + np.arange(self.action_size)) if full_covariance else 2 * action_size
        )
        self.layer_norm = torch.nn.LayerNorm(input_size) if layer_norm else None
        self.v_head = MlpModel(
            input_size=self.shared_features_dim,
            hidden_sizes=[256, ],
            output_size=1 if linear_value_output else None,
        )

    def forward(self, observation, prev_action, prev_reward, init_rnn_state=None):
        lead_dim, T, B, _ = infer_leading_dims(observation, self._obs_ndim)
        assert not torch.any(torch.isnan(observation)), 'obs elem is nan'

        obs_flat = observation.reshape(T * B, -1)
        if self.layer_norm:
            obs_flat = torch.tanh(self.layer_norm(obs_flat))
        features = self.shared_mlp(obs_flat)
        action = self.mu_head(features)

        v = self.v_head(features).squeeze(-1)

        if self.full_covariance:
            mu, lower_triag_cov = (action[:, :self.action_size], action[:, self.action_size:])
            upper_triangular_indeces = torch.triu_indices(self.action_size, self.action_size)  # .transpose(-1, -2)
            covariance = torch.zeros((T * B, self.action_size, self.action_size), device=mu.device)
            covariance[:, upper_triangular_indeces[0], upper_triangular_indeces[1]] = lower_triag_cov
            covariance = covariance.transpose(-2, -1)  # make lower triangular matrix
            diagonal = torch.diagonal(covariance, dim1=-2, dim2=-1)
            diag_indeces = np.diag_indices(self.action_size)
            diagonal = torch.log(1 + torch.exp(diagonal))  # +  1e-5)
            covariance[:, diag_indeces[0], diag_indeces[1]] = diagonal
        else:
            mu, covariance = (action[:, :self.action_size], action[:, self.action_size:])
            covariance = torch.log(1 + torch.exp(covariance)) # softplus

        # Restore leading dimensions: [T,B], [B], or [], as input.
        # mu = torch.tanh(mu)
        mu, covariance, v = restore_leading_dims((mu, covariance, v), lead_dim, T, B)

        return mu, covariance, v, torch.zeros(1, B, 1)  # return fake rnn state


class TransformerModel(torch.nn.Module):
    def __init__(self,
                 observation_shape,
                 action_size,
                 max_episode_length=150,
                 hidden_sizes=None,  # None for default (see below).
                 init_log_std=0.,
                 normalize_observation=False,
                 linear_value_output=True,
                 norm_obs_clip=10,
                 norm_obs_var_clip=1e-6,
                 size='medium',
                 **kwargs
                 ):
        super().__init__()
        self.state_size = np.prod(observation_shape.state)
        self.action_size = action_size
        self.normalize_observation = normalize_observation
        self.transformer = DemonstrationTransformerModel(input_size=self.state_size,
                                                         output_size=SIZES[size]['d_model'],
                                                         demonstration_length=max_episode_length,
                                                         size=size,
                                                         **kwargs)
        self.pi_head = MlpModel(input_size=self.transformer.d_model,
                                hidden_sizes=[256, ],
                                output_size=2 * self.action_size)
        self.value_head = MlpModel(input_size=self.transformer.d_model,
                                   hidden_sizes=[256, ],
                                   output_size=1 if linear_value_output else None)

    def forward(self, observation, prev_action, prev_reward, rnn_state):
        lead_dim, T, B, _ = infer_leading_dims(observation.state, 1)
        transformer_output, rnn_state = self.transformer(observation, prev_action, prev_reward, rnn_state)
        pi_output = self.pi_head(transformer_output).view(T * B, -1)
        mu, log_std = pi_output[:, :self.action_size], pi_output[:, self.action_size:]
        log_std = torch.log(1 + torch.exp(log_std))
        # covariance = torch.log(1 + torch.exp(log_std))  # softplus
        # covariance = torch.log(covariance) # agent expects loc_std
        # covariance = torch.diag_embed(covariance)

        value = self.value_head(transformer_output).reshape(T * B, -1)

        mu, log_std, value = restore_leading_dims((mu, log_std, value), lead_dim, T, B)

        return mu, log_std, value, rnn_state

    def update_obs_rms(self, observation):
        if self.normalize_observation:
            self.obs_rms.update(observation)


class GeneralizedTransformerModel(torch.nn.Module):
    def __init__(self,
                 observation_shape,
                 action_size,
                 episode_length=50,
                 demonstration_length=25,
                 normalize_observation=False,
                 linear_value_output=True,
                 size='medium',
                 layer_norm=True,
                 seperate_value_network=True,
                 **kwargs
                 ):
        super().__init__()
        self.episode_length = episode_length
        self.state_size = np.prod(observation_shape.state)
        self.action_size = action_size
        self.normalize_observation = normalize_observation
        self.d_model = SIZES[size]['d_model']
        self.layer_norm = torch.nn.LayerNorm(self.state_size) if layer_norm else None
        self.observation_size = self.state_size
        self.encoding_layer = torch.nn.Linear(self.observation_size, self.d_model)
        self.transformer = GatedTransformer(**SIZES[size], only_encoder=True)
        self.value_transformer = GatedTransformer(**SIZES[size], only_encoder=True) if seperate_value_network else None
        self.pi_head = MlpModel(input_size=self.d_model,
                                hidden_sizes=[256, ],
                                output_size=2 * self.action_size)
        self.value_head = MlpModel(input_size=self.d_model,
                                   hidden_sizes=[256, ],
                                   output_size=1 if linear_value_output else None)
        self.subsequent_mask = generate_square_subsequent_mask(demonstration_length + episode_length)

    def forward(self, observation, prev_action, prev_reward, rnn_state):
        lead_dim, T, B, _ = infer_leading_dims(observation.state, 1)
        demonstration = observation.demonstration.view(T, B, -1, self.state_size)[0].permute(1, 0, 2)
        observation = observation.state.view(T, B, -1)
        demonstration_length = demonstration.shape[0]
        self.subsequent_mask = self.subsequent_mask.to(observation.device)

        if T == 1:
            # non differentiable sampling
            if rnn_state is None:
                observations = torch.zeros((self.episode_length, B, self.observation_size), device=observation.device)
                length = torch.zeros((1, B, 1), device=observation.device, dtype=torch.int64)
            else:
                observations = rnn_state.observations
                length = rnn_state.length

            observations[length[0, :, 0], torch.arange(B)] = observation
            input_sequence = torch.cat((demonstration, observations), dim=0)
            if self.layer_norm:
                input_sequence = torch.tanh(self.layer_norm(input_sequence))
            # subsequent_mask = generate_square_subsequent_mask(input_sequence.shape[0], device=input_sequence.device)
            transformer_output = self.transformer(self.encoding_layer(input_sequence), src_mask=self.subsequent_mask)
            output = transformer_output[length[0, :, 0] + demonstration_length, torch.arange(B)]
            gaussian_parameters = self.pi_head(output)
            mu, std = (gaussian_parameters[:, :self.action_size], gaussian_parameters[:, self.action_size:])
            value = None

            rnn_state = RnnState(observations=observations, length=length + 1)
        else:
            input_sequence = torch.cat((demonstration, observation), dim=0)
            if self.layer_norm:
                input_sequence = torch.tanh(self.layer_norm(input_sequence))
            # subsequent_mask = generate_square_subsequent_mask(input_sequence.shape[0], device=observation.device)
            transformer_output = self.transformer(self.encoding_layer(input_sequence), src_mask=self.subsequent_mask)
            transformer_output = transformer_output[demonstration_length:].reshape(T * B, -1)
            gaussian_parameters = self.pi_head(transformer_output)
            mu, std = (gaussian_parameters[:, :self.action_size], gaussian_parameters[:, self.action_size:])
            if self.value_transformer is not None:
                value = self.value_transformer(self.encoding_layer(input_sequence), src_mask=self.subsequent_mask)
                value = value[demonstration_length:].reshape(T * B, -1)
            else:
                value = transformer_output
            value = self.value_head(value)

            rnn_state = None

        std = torch.log(1 + torch.exp(std))
        mu = torch.tanh(mu)
        mu, std = restore_leading_dims((mu, std), lead_dim, T, B)
        value = restore_leading_dims(value, lead_dim, T, B) if value is not None else value
        return mu, std, value, rnn_state

    def update_obs_rms(self, observation):
        if self.normalize_observation:
            self.obs_rms.update(observation)


class CategoricalFfModel(torch.nn.Module):
    """
    Model commonly used in Mujoco locomotion agents: an MLP which outputs
    distribution means, separate parameter for learned log_std, and separate
    MLP for state-value estimate.
    """

    def __init__(
            self,
            observation_shape,
            action_size,
            hidden_sizes=None,  # None for default (see below).
            init_log_std=0.,
            normalize_observation=False,
            norm_obs_clip=10,
            norm_obs_var_clip=1e-6,
    ):
        """Instantiate neural net modules according to inputs."""
        super().__init__()
        self._obs_ndim = len(observation_shape)
        input_size = int(np.prod(observation_shape))
        hidden_sizes = hidden_sizes or [256, 256]
        self.action_size = action_size
        self.mu_mlp = MlpModel(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=action_size,
        )
        self.v = MlpModel(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=1,
        )
        self.log_std = torch.nn.Parameter(init_log_std * torch.ones(action_size))
        if normalize_observation:
            self.obs_rms = RunningMeanStdModel(observation_shape)
            self.norm_obs_clip = norm_obs_clip
            self.norm_obs_var_clip = norm_obs_var_clip
        self.normalize_observation = normalize_observation

    def forward(self, observation, prev_action, prev_reward):
        """
        Compute mean, log_std, and value estimate from input state. Infers
        leading dimensions of input: can be [T,B], [B], or []; provides
        returns with same leading dims.  Intermediate feedforward layers
        process as [T*B,H], with T=1,B=1 when not given. Used both in sampler
        and in algorithm (both via the agent).
        """
        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, _ = infer_leading_dims(observation, self._obs_ndim)
        assert not torch.any(torch.isnan(observation)), 'obs elem is nan'

        if self.normalize_observation:
            obs_var = self.obs_rms.var
            if self.norm_obs_var_clip is not None:
                obs_var = torch.clamp(obs_var, min=self.norm_obs_var_clip)
            observation = torch.clamp((observation - self.obs_rms.mean) /
                                      obs_var.sqrt(), -self.norm_obs_clip, self.norm_obs_clip)

        obs_flat = observation.view(T * B, -1)
        action = torch.softmax(self.mu_mlp(obs_flat), dim=-1)
        v = self.v(obs_flat).squeeze(-1)

        # Restore leading dimensions: [T,B], [B], or [], as input.
        action, v = restore_leading_dims((action, v), lead_dim, T, B)

        return action, v

    def update_obs_rms(self, observation):
        if self.normalize_observation:
            self.obs_rms.update(observation)
