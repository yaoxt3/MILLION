import torch
import numpy as np
from torch import relu
from torch.nn import LSTM, Linear
from learning_to_be_taught.models.attention_models import AttentionModel
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.models.mlp import MlpModel
from rlpyt.utils.collections import namedarraytuple

RnnState = namedarraytuple("RnnState", ["h", "c"])

class PiLstmDemonstrationModel(torch.nn.Module):
    def __init__(self, observation_shape, action_size):
        super().__init__()
        input_size = observation_shape.state[0]
        self.demonstration_length = observation_shape.demonstration[0]
        self.action_size = action_size
        self.lstm_size = 4 * input_size
        self.demonstration_LSTM = LSTM(input_size, hidden_size=self.lstm_size)
        self.combination_layer = torch.nn.Linear(input_size + self.lstm_size, 2 * self.lstm_size)
        self.lstm_layer = torch.nn.LSTM(2 * self.lstm_size, self.lstm_size)
        self.output_mlp = MlpModel(input_size=self.lstm_size,
                                   hidden_sizes=[256, 256],
                                   output_size=2 * action_size)

    def forward(self, observation, prev_action, prev_reward, rnn_state):
        lead_dim, T, B, _ = infer_leading_dims(observation.state, 1)
        current_obs = observation.state.view(T, B, -1)
        demonstration = observation.demonstration.view(T, B, self.demonstration_length, -1)[0]
        demonstration = demonstration.permute(1, 0, 2)  # now demonstration time steps x Batch x features
        lstm_out, _ = self.demonstration_LSTM(demonstration, None)
        demonstration_encoding = lstm_out[-1]  # now in shape batch x features

        x = relu(self.combination_layer(torch.cat((current_obs, demonstration_encoding.expand(T, B, -1)), dim=-1)))
        rnn_state = None if rnn_state is None else tuple(rnn_state)
        lstm_out, (hidden_state, cell_state) = self.lstm_layer(x, rnn_state)
        output = self.output_mlp(lstm_out.view(T * B, -1))
        mu, log_std = output[:, :self.action_size], output[:, self.action_size:]
        mu, log_std = restore_leading_dims((mu, log_std), lead_dim, T, B)
        rnn_state = RnnState(h=hidden_state, c=cell_state)
        return mu, log_std, rnn_state

class QLstmDemonstrationModel(torch.nn.Module):
    def __init__(self, observation_shape, action_size):
        super().__init__()
        input_size = observation_shape.state[0]
        self.demonstration_length = observation_shape.demonstration[0]
        self.lstm_size = 4 * input_size
        self.demonstration_LSTM = LSTM(input_size, hidden_size=self.lstm_size)
        self.combination_layer = torch.nn.Linear(input_size + self.lstm_size, 2 * self.lstm_size)
        self.lstm_layer = torch.nn.LSTM(2 * self.lstm_size, self.lstm_size)
        self.output_mlp = MlpModel(input_size=self.lstm_size + action_size,
                                   hidden_sizes=[256, 256],
                                   output_size=1)

    def forward(self, observation, prev_action, prev_reward, action, rnn_state):
        lead_dim, T, B, _ = infer_leading_dims(observation.state, 1)
        current_obs = observation.state.view(T, B, -1)
        action = action.view(T, B, -1)
        demonstration = observation.demonstration.view(T, B, self.demonstration_length, -1)[0]
        demonstration = demonstration.permute(1, 0, 2)  # now demonstration time steps x Batch x features
        lstm_out, _ = self.demonstration_LSTM(demonstration, None)
        demonstration_encoding = lstm_out[-1]  # now in shape batch x features

        x = relu(self.combination_layer(torch.cat((current_obs, demonstration_encoding.expand(T, B, -1)), dim=-1)))
        rnn_state = None if rnn_state is None else tuple(rnn_state)
        lstm_out, (hidden_state, cell_state) = self.lstm_layer(x, rnn_state)
        output = self.output_mlp(torch.cat((lstm_out, action), dim=-1))

        # Restore leading dimensions: [T,B], [B], or [], as input.
        output = restore_leading_dims(output.view(T * B, -1), lead_dim, T, B).squeeze(-1)
        # Model should always leave B-dimension in rnn state: [N,B,H].
        rnn_state = RnnState(h=hidden_state, c=cell_state)
        return output, rnn_state
