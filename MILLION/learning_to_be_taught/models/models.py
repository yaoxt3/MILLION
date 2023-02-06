from rlpyt.models.mlp import MlpModel
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
import torch
from torch import relu
from torch.nn import Linear, LSTMCell, LSTM
from rlpyt.utils.collections import namedarraytuple

RnnState = namedarraytuple("RnnState", ["h", "c"])


class SimpleQModel(torch.nn.Module):
    def __init__(self, input_size=9, hidden_sizes=[32, ], action_dim=4):
        super().__init__()
        self.mlp = MlpModel(input_size=input_size, hidden_sizes=hidden_sizes)
        self.output_layer = torch.nn.Linear(32, action_dim)

    def forward(self, observation, prev_action, prev_reward):
        x = self.mlp(observation.float())
        return self.output_layer(x)


class GoalObsQModel(torch.nn.Module):
    def __init__(self, env_spaces):
        super().__init__()
        input_size = env_spaces.observation.shape.state[0]
        action_dim = env_spaces.action.n
        self.demonstration_length = env_spaces.observation.shape.demonstration[0]
        self.combination_layer = Linear(2 * input_size, 64)
        self.output_layer = torch.nn.Linear(64, action_dim)

    def forward(self, observation, prev_action, prev_reward):  # , rnn_state):
        lead_dim, T, B, _ = infer_leading_dims(observation.state, 1)
        demonstration = observation.demonstration.view(T, B, self.demonstration_length, -1)
        current_obs = observation.state.view(T, B, -1)

        x = relu(self.combination_layer(torch.cat((current_obs, demonstration[:, :, -1]), dim=-1)))
        q = self.output_layer(x)

        # Restore leading dimensions: [T,B], [B], or [], as input.
        output = restore_leading_dims(q.reshape(T * B, -1), lead_dim, T, B)
        # Model should always leave B-dimension in rnn state: [N,B,H].
        rnn_state = RnnState(h=torch.zeros(1, B, 1), c=torch.zeros(1, B, 1))
        return output  # , rnn_state


class DemonstrationQModel(torch.nn.Module):
    def __init__(self, env_spaces):
        super().__init__()
        input_size = env_spaces.observation.shape.state[0]
        action_dim = env_spaces.action.n
        self.demonstration_length = env_spaces.observation.shape.demonstration[0]
        self.mlp = MlpModel(input_size + self.demonstration_length * input_size, hidden_sizes=[4 * input_size, ])
        self.output_layer = Linear(4 * input_size, action_dim)

    def forward(self, observation, prev_action, prev_reward):  # , rnn_state):
        lead_dim, T, B, _ = infer_leading_dims(observation.state, 1)
        current_obs = observation.state.view(T * B, -1)
        demonstration = observation.demonstration.view(T * B, -1)
        output = self.mlp(torch.cat((current_obs, demonstration), dim=-1))
        output = self.output_layer(output)
        output = restore_leading_dims(output, lead_dim, T, B)
        # Model should always leave B-dimension in rnn state: [N,B,H].
        # rnn_state = RnnState(h=torch.zeros(1, B, 1), c=torch.zeros(1, B, 1))
        return output  # , rnn_state


class DemonstrationRecurrentQModel(torch.nn.Module):
    def __init__(self, env_spaces):
        super().__init__()
        input_size = env_spaces.observation.shape.state[0]
        action_dim = env_spaces.action.n
        self.demonstration_length = env_spaces.observation.shape.demonstration[0]
        self.lstm_size = 4 * input_size
        self.demonstration_LSTM = LSTM(input_size, hidden_size=self.lstm_size)
        self.combination_layer = torch.nn.Linear(input_size + self.lstm_size, 64)
        self.middle_layer = Linear(64, 64)
        self.output_layer = torch.nn.Linear(64, action_dim)

    def forward(self, observation, prev_action, prev_reward):  # , rnn_state):
        lead_dim, T, B, _ = infer_leading_dims(observation.state, 1)
        current_obs = observation.state.view(T, B, -1)
        demonstration = observation.demonstration.view(T, B, self.demonstration_length, -1)[0]
        demonstration = demonstration.permute(1, 0, 2)  # now demonstration time steps x Batch x features
        lstm_out, rnn_state = self.demonstration_LSTM(demonstration, None)
        demonstration_encoding = lstm_out[-1]  # now in shape batch x features

        x = relu(self.combination_layer(torch.cat((current_obs, demonstration_encoding.expand(T, B, -1)), dim=-1)))
        x = relu(self.middle_layer(x))
        output = self.output_layer(x).view(T * B, -1)

        # Restore leading dimensions: [T,B], [B], or [], as input.
        output = restore_leading_dims(output, lead_dim, T, B)
        # Model should always leave B-dimension in rnn state: [N,B,H].
        # rnn_state = RnnState(h=torch.zeros(1, B, 1), c=torch.zeros(1, B, 1))
        return output  # , rnn_state
