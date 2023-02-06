import torch
import numpy as np
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.models.mlp import MlpModel
from rlpyt.utils.collections import namedarraytuple

RnnState = namedarraytuple("RnnState", ["h", "c"])

class LstmModel(torch.nn.Module):
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
            lstm_size=128,
            lstm_layers=2
    ):
        """Instantiate neural net modules according to inputs."""
        super().__init__()
        self._obs_ndim = len(observation_shape.state)
        input_size = int(np.prod(observation_shape.state))
        self.full_covariance = full_covariance
        self.action_size = action_size
        self.shared_features_dim = 512
        self.shared_mlp = MlpModel(
            input_size=input_size,
            hidden_sizes=[512, self.shared_features_dim]
        )
        self.lstm = torch.nn.LSTM(self.shared_features_dim, lstm_size, lstm_layers)

        self.mu_head = MlpModel(
            input_size=lstm_size,
            hidden_sizes=[256],
            output_size=action_size + np.sum(1 + np.arange(self.action_size)) if full_covariance else 2 * action_size
        )
        self.layer_norm = torch.nn.LayerNorm(input_size) if layer_norm else None
        self.v_head = MlpModel(
            input_size=lstm_size,
            hidden_sizes=[256, ],
            output_size=1 if linear_value_output else None,
        )

    def forward(self, observation, prev_action, prev_reward, rnn_state=None):
        observation = observation.state
        lead_dim, T, B, _ = infer_leading_dims(observation, self._obs_ndim)
        rnn_state = None if rnn_state is None else (rnn_state.h.contiguous(), rnn_state.c.contiguous())
        assert not torch.any(torch.isnan(observation)), 'obs elem is nan'

        obs_flat = observation.reshape(T, B, -1)
        if self.layer_norm:
            obs_flat = torch.tanh(self.layer_norm(obs_flat))

        features = self.shared_mlp(obs_flat)
        features, (hidden_state, cell_state) = self.lstm(features, rnn_state)

        features = features.reshape(T * B, -1)
        action = self.mu_head(features)
        v = self.v_head(features).squeeze(-1)

        mu, covariance = (action[:, :self.action_size], action[:, self.action_size:])
        covariance = torch.log(1 + torch.exp(covariance)) # softplus

        # Restore leading dimensions: [T,B], [B], or [], as input.
        mu = torch.tanh(mu)
        mu, covariance, v = restore_leading_dims((mu, covariance, v), lead_dim, T, B)
        rnn_state = RnnState(h=hidden_state, c=cell_state)

        return mu, covariance, v, rnn_state

