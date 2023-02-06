import torch
import numpy as np
from learning_to_be_taught.models.attention_models import AttentionModel
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.models.mlp import MlpModel


class FakeRecurrentPiModel(torch.nn.Module):
    """Action distrubition MLP model for SAC agent."""

    def __init__(
            self,
            observation_shape,
            action_size,
            hidden_sizes=None,
    ):
        super().__init__()
        self._obs_ndim = len(observation_shape.state)
        self._action_size = action_size
        hidden_sizes = [256, 256] if hidden_sizes is None else hidden_sizes
        self.mlp = MlpModel(
            input_size=int(np.prod(observation_shape.state)),
            hidden_sizes=hidden_sizes,
            output_size=action_size * 2,
        )

    def forward(self, observation, prev_action, prev_reward, rnn_state):
        lead_dim, T, B, _ = infer_leading_dims(observation.state, self._obs_ndim)
        output = self.mlp(observation.state.view(T * B, -1))
        mu, log_std = output[:, :self._action_size], output[:, self._action_size:]
        mu, log_std = restore_leading_dims((mu, log_std), lead_dim, T, B)
        fake_rnn_state = torch.zeros(1, B, 1)
        return mu, log_std, fake_rnn_state


class FakeRecurrentQModel(torch.nn.Module):
    """Q portion of the model for DDPG, an MLP."""

    def __init__(
            self,
            observation_shape,
            action_size,
            hidden_sizes=None,
    ):
        """Instantiate neural net according to inputs."""
        super().__init__()
        hidden_sizes = [256, 256] if hidden_sizes is None else hidden_sizes
        self.mlp = MlpModel(
            input_size=np.prod(observation_shape.state) + action_size,
            hidden_sizes=hidden_sizes,
            output_size=1,
        )

    def forward(self, observation, prev_action, prev_reward, action, rnn_state):
        lead_dim, T, B, _ = infer_leading_dims(observation.state, 1)
        if action is not None:
            q_input = torch.cat([observation.state.view(T * B, -1), action.view(T * B, -1)], dim=1)
            q = self.mlp(q_input).squeeze(-1)
        else:
            q = torch.zeros(T * B)

        q = restore_leading_dims(q, lead_dim, T, B)
        fake_rnn_state = torch.zeros(1, B, 1)
        return q, fake_rnn_state
