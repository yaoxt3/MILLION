import torch
import time
import numpy as np
# from compressive_transformer_pytorch import CompressiveTransformer as CompressiveTransformerPyTorch
# from compressive_transformer_pytorch import CompressiveTransformer as CompressiveTransformerPyTorch
from learning_to_be_taught.models.compressive_transformer_pytorch import CompressiveTransformerPyTorch
from learning_to_be_taught.models.compressive_transformer_pytorch import Memory
from rlpyt.models.mlp import MlpModel
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.utils.collections import namedarraytuple
# from learning_to_be_taught.models.gated_transformer import GatedTransformer, SIZES
from learning_to_be_taught.models.transformer_models import generate_square_subsequent_mask


State = namedarraytuple("State", ["sequence", "memory", 'compressed_memory', 'length'])#, 'memory_filled', 'compressed_memory_filled'])

SIZES = dict(
    zero=dict(dim=64,
              cmem_ratio=4,
              num_heads=4,
              depth=0),
    tiny=dict(dim=64,
              cmem_ratio=4,
              num_heads=4,
              depth=1),
    small=dict(dim=64,
               cmem_ratio=4,
               num_heads=4,
               depth=3),
    medium=dict(dim=64,
                cmem_ratio=4,
                num_heads=4,
                depth=6),
    large=dict(dim=1024,
               cmem_ratio=4,
               num_heads=12,
               depth=12)
)


class CompressiveTransformer(torch.nn.Module):
    def __init__(self,
                 observation_shape,
                 action_size,
                 linear_value_output=True,
                 sequence_length=64,
                 seperate_value_network=False,
                 observation_normalization=True,
                 size='medium',
                 ):
        super().__init__()
        self.state_size = np.prod(observation_shape.state)
        self.action_size = action_size
        self.sequence_length = sequence_length
        self.transformer_dim = SIZES[size]['dim']
        self.depth = SIZES[size]['depth']
        self.cmem_ratio = SIZES[size]['cmem_ratio']
        self.cmem_length = self.sequence_length // self.cmem_ratio
        memory_layers = range(1, self.depth + 1)
        self.transformer = CompressiveTransformerPyTorch(
            num_tokens=20000,
            emb_dim=self.state_size,  # embedding dimensions, embedding factorization from Albert paper
            dim=self.transformer_dim,
            heads=SIZES[size]['num_heads'],
            depth=self.depth,
            seq_len=self.sequence_length,
            mem_len=self.sequence_length,  # memory length
            # cmem_len=self.cmem_length,  # compressed memory buffer length
            # cmem_ratio=self.cmem_ratio,  # compressed memory ratio, 4 was recommended in paper
            reconstruction_loss_weight=1,  # weight to place on compressed memory reconstruction loss
            gru_gated_residual=True,
            # whether to gate the residual intersection, from 'Stabilizing Transformer for RL' paper
            memory_layers=memory_layers,
        )
        self.transformer.token_emb = torch.nn.Identity()  # don't use token embedding in compressive transforrmer
        self.transformer.to_logits = torch.nn.Identity()
        
        if observation_normalization:
            self.input_layer_norm = torch.nn.LayerNorm(self.state_size)
        else:
            self.input_layer_norm = torch.nn.Identity()
        # self.output_layer_norm = torch.nn.LayerNorm(self.transformer_dim)
        self.output_layer_norm = torch.nn.Identity()
        self.softplus = torch.nn.Softplus()
        self.pi_head = MlpModel(input_size=self.transformer_dim,
                                hidden_sizes=[256, ],
                                output_size=2 * action_size)
        self.value_head = MlpModel(input_size=self.transformer_dim,
                                   hidden_sizes=[256, ],
                                   output_size=1 if linear_value_output else None)
        self.mask = torch.ones((self.sequence_length, self.sequence_length), dtype=torch.int8).triu()

    def forward(self, observation, prev_action, prev_reward, state):
        lead_dim, T, B, _ = infer_leading_dims(observation.state, 1)
        # print(f'step in episode {observation.step_in_episode}')
        aux_loss = None
        if T == 1:
            transformer_output, state = self.sample_forward(observation.state, state)
            value = torch.zeros(B)
        elif T == self.sequence_length:
            transformer_output, aux_loss = self.optim_forward(observation.state, state)
            value = self.value_head(transformer_output).reshape(T * B, -1)
        else:
            raise NotImplementedError

        pi_output = self.pi_head(transformer_output).view(T * B, -1)
        mu, std = pi_output[:, :self.action_size], pi_output[:, self.action_size:]
        std = self.softplus(std)
        mu = torch.tanh(mu)

        # pi_output = self.softplus(pi_output * 1) + 1
        # mu, std = pi_output[:, :self.action_size], pi_output[:, self.action_size:]


        mu, std, value = restore_leading_dims((mu, std, value), lead_dim, T, B)
        return mu, std, value, state, aux_loss

    def sample_forward(self, observation, state):
        lead_dim, T, B, _ = infer_leading_dims(observation, 1)
        observation = self.input_layer_norm(observation)

        device = observation.device
        if state is None:
            observations = torch.zeros((self.sequence_length, B, self.state_size), device=device)
            length = torch.zeros((1, B, 1), device=device, dtype=torch.int64)
            memory = torch.zeros((self.depth, B, self.sequence_length, self.transformer_dim), device=device)
            compressed_memory = torch.zeros((self.depth, B, self.cmem_length, self.transformer_dim), device=device)
        else:
            observations = state.sequence
            length = state.length.clamp_max(self.sequence_length - 1)
            memory = state.memory
            compressed_memory = state.compressed_memory

        # write new observations in tensor with older observations
        observation = observation.view(B, -1)
        # print(f'observations shape {observations.shape} observation {observation.shape}')
        # observations = torch.cat((observations, observation), dim=0)[1:]
        indexes = tuple(torch.cat((length[0, :], torch.arange(B, device=device).unsqueeze(-1)), dim=-1).t())
        observations.index_put_(indexes, observation)

        transformer_output, new_memory, _ = self.transformer(observations.transpose(0, 1), Memory(mem=memory, compressed_mem=None))
        transformer_output = self.output_layer_norm(transformer_output).transpose(0,1)
        # output = transformer_output.transpose(0, 1)[-1]
        output = transformer_output[length[0, :, 0], torch.arange(B)]
        # output = transformer_output[-1].reshape(T, B, -1)
        length = torch.fmod(length + 1, self.sequence_length)

        reset = (length == 0).int()[0, :, 0].reshape(B, 1, 1, 1).transpose(0, 1).expand_as(memory)
        # print(f'length {length[0, :, 0]}')
        # if B > 1 and length[0, 0, 0] != length[0, 1, 0]:
        #     breakpoint()
        memory = reset * new_memory.mem + (1 - reset) * memory
        # memory = new_memory.mem

        state = State(sequence=observations, length=length, memory=memory, compressed_memory=compressed_memory)
        return output, state

    def optim_forward(self, observation, state):
        observation = self.input_layer_norm(observation)
        output, _, aux_loss = self.transformer(observation.transpose(0, 1), Memory(mem=state.memory, compressed_mem=None))
        output = self.output_layer_norm(output)
        return output.transpose(0, 1), aux_loss

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
            hidden_sizes=None,  # None for default (see below).
            init_log_std=0.,
            normalize_observation=False,
            linear_value_output=True,
            norm_obs_clip=10,
            full_covariance=False,
            norm_obs_var_clip=1e-6,
    ):
        """Instantiate neural net modules according to inputs."""
        super().__init__()
        self._obs_ndim = len(observation_shape.state)
        input_size = int(np.prod(observation_shape.state))
        self.full_covariance = full_covariance
        hidden_sizes = hidden_sizes or [256, 256]
        self.action_size = action_size
        self.shared_features_dim = 256
        self.softplus = torch.nn.Softplus()
        self.shared_mlp = MlpModel(
            input_size=input_size,
            hidden_sizes=[512, self.shared_features_dim]
        )
        self.mu_head = MlpModel(
            input_size=input_size,
            hidden_sizes=[256, 256],
            # output_size=action_size * 2,
            output_size=action_size + np.sum(1 + np.arange(self.action_size)) if full_covariance else 2 * action_size
        )
        self.layer_norm = torch.nn.LayerNorm(input_size)
        # list(self.mu_head.parameters())[-1].data = list(self.mu_head.parameters())[-1].data / 100
        # list(self.mu_head.parameters())[-2].data = list(self.mu_head.parameters())[-2].data / 100
        self.v_head = MlpModel(
            input_size=input_size,
            hidden_sizes=[256, 256],
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
        lead_dim, T, B, _ = infer_leading_dims(observation.state, self._obs_ndim)
        assert not torch.any(torch.isnan(observation.state)), 'obs elem is nan'

        obs_flat = observation.state.reshape(T * B, -1)
        # obs_flat = self.layer_norm(obs_flat)
        # features = self.shared_mlp(obs_flat)
        action = self.mu_head(obs_flat)

        v = self.v_head(obs_flat).squeeze(-1)

        # mu, std = (action[:, :self.action_size], action[:, self.action_size:])
        # std = self.softplus(std)

        pi_output = self.softplus(action * 8) + 1
        mu, std = pi_output[:, :self.action_size], pi_output[:, self.action_size:]

        # Restore leading dimensions: [T,B], [B], or [], as input.
        mu, std, v = restore_leading_dims((mu, std, v), lead_dim, T, B)

        return mu, std, v, torch.zeros(1, B, 1), None  # return fake rnn state
