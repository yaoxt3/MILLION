from rlpyt.models.mlp import MlpModel
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
import torch
from torch import relu
from torch.nn import Linear, LSTMCell, LSTM
import numpy as np
from rlpyt.utils.collections import namedarraytuple

RnnState = namedarraytuple("RnnState", ["h", "c"])


class RecurrentPiModel(torch.nn.Module):
    """Action distrubition MLP model for SAC agent."""

    def __init__(
            self,
            observation_shape,
            hidden_sizes,
            action_size,
    ):
        super().__init__()
        self._obs_ndim = len(observation_shape)
        self._action_size = action_size
        self.lstm_size = 64
        hidden_sizes = [64, 64] if hidden_sizes is None else hidden_sizes
        self.mlp = MlpModel(
            input_size=int(np.prod(observation_shape)),
            hidden_sizes=hidden_sizes,
            output_size=self.lstm_size * 2,
        )
        self.lstm = torch.nn.LSTM(2 * self.lstm_size, self.lstm_size)
        self.output_layer = torch.nn.Linear(self.lstm_size, action_size * 2)

    def forward(self, observation, prev_action, prev_reward, rnn_state):
        lead_dim, T, B, _ = infer_leading_dims(observation, self._obs_ndim)
        rnn_state = None if rnn_state is None else tuple(rnn_state)

        x = self.mlp(observation.view(T, B, -1))
        x, (hidden_state, cell_state) = self.lstm(x, rnn_state)
        output = self.output_layer(x.view(T * B, -1))

        mu, log_std = output[:, :self._action_size], output[:, self._action_size:]
        mu, log_std = restore_leading_dims((mu, log_std), lead_dim, T, B)
        rnn_state = RnnState(h=hidden_state, c=cell_state)
        return mu, log_std, rnn_state


class RecurrentQModel(torch.nn.Module):
    """Action distrubition MLP model for SAC agent."""

    def __init__(
            self,
            observation_shape,
            hidden_sizes,
            action_size,
    ):
        super().__init__()
        self._obs_ndim = len(observation_shape)
        self._action_size = action_size
        self.lstm_size = 64
        hidden_sizes = [64, 64] if hidden_sizes is None else hidden_sizes
        self.mlp = MlpModel(
            input_size=np.prod(observation_shape),
            hidden_sizes=hidden_sizes,
            output_size=self.lstm_size * 2,
        )
        self.lstm = torch.nn.LSTM(2 * self.lstm_size, self.lstm_size)
        self.output_mlp = MlpModel(
            input_size=self.lstm_size + action_size,
            hidden_sizes=[256],
            output_size=1
        )

    def forward(self, observation, prev_action, prev_reward, action, rnn_state):
        lead_dim, T, B, _ = infer_leading_dims(observation, self._obs_ndim)
        rnn_state = None if rnn_state is None else tuple(rnn_state)

        q_input = observation.view(T * B, -1)
        x = self.mlp(q_input)
        x, (hidden_state, cell_state) = self.lstm(x.view(T, B, -1), rnn_state)
        x = x.view(T * B, -1)
        if action is not None:
            output = self.output_mlp(torch.cat((x, action.view(T * B, -1)), dim=-1)).squeeze(-1)
        else:
            output = torch.zeros(T * B)

        rnn_state = RnnState(h=hidden_state, c=cell_state)
        output = restore_leading_dims(output, lead_dim, T, B)
        return output, rnn_state

