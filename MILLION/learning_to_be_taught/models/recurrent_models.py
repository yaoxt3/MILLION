from rlpyt.models.mlp import MlpModel
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
import torch
from torch import relu
from torch.nn import Linear, LSTMCell, LSTM
from rlpyt.utils.collections import namedarraytuple

RnnState = namedarraytuple("RnnState", ["h", "c"])


class DemonstrationRecurrentQModel(torch.nn.Module):
    def __init__(self, env_spaces):
        super().__init__()
        input_size = env_spaces.observation.shape.state[0]
        action_dim = env_spaces.action.n
        self.demonstration_length = env_spaces.observation.shape.demonstration[0]
        self.lstm_size = 4 * input_size
        self.demonstration_LSTM = LSTM(input_size, hidden_size=self.lstm_size)
        self.combination_layer = torch.nn.Linear(input_size + self.lstm_size, 2 * self.lstm_size)
        self.middle_layer = Linear(2 * self.lstm_size, 2 * self.lstm_size)
        self.output_layer = torch.nn.Linear(2 * self.lstm_size, action_dim)

    def forward(self, observation, prev_action, prev_reward, rnn_state):
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
        rnn_state = RnnState(h=torch.zeros(1, B, 1), c=torch.zeros(1, B, 1))
        return output, rnn_state


class FakeRecurrentGoalObsQModel(torch.nn.Module):
    """feed forward q model that works with recurrent dqn algo"""

    def __init__(self, env_spaces):
        super().__init__()
        input_size = env_spaces.observation.shape.state[0]
        action_dim = env_spaces.action.n
        self.demonstration_length = env_spaces.observation.shape.demonstration[0]
        self.combination_layer = Linear(2 * input_size, 128)
        self.middle_layer = Linear(128, 128)
        self.output_layer = torch.nn.Linear(128, action_dim)

    def forward(self, observation, prev_action, prev_reward, rnn_state):
        lead_dim, T, B, _ = infer_leading_dims(observation.state, 1)
        demonstration = observation.demonstration.view(T, B, self.demonstration_length, -1)
        current_obs = observation.state.view(T, B, -1)

        x = relu(self.combination_layer(torch.cat((current_obs, demonstration[:, :, -1]), dim=-1)))
        x = relu(self.middle_layer(x))
        q = self.output_layer(x)

        # Restore leading dimensions: [T,B], [B], or [], as input.
        output = restore_leading_dims(q.reshape(T * B, -1), lead_dim, T, B)
        # Model should always leave B-dimension in rnn state: [N,B,H].
        rnn_state = RnnState(h=torch.zeros(1, B, 1), c=torch.zeros(1, B, 1))
        return output, rnn_state


class RecurrentGoalObsQModel(torch.nn.Module):
    def __init__(self, env_spaces):
        super().__init__()
        self.demonstration_length = env_spaces.observation.shape.demonstration[0]
        input_size = env_spaces.observation.shape.state[0]
        self.demonstration_feature_layers = torch.nn.ModuleList()
        for i in range(self.demonstration_length):
            self.demonstration_feature_layers.append(torch.nn.Linear(input_size, 16))

        self.current_obs_layer = torch.nn.Linear(input_size, 32)
        self.goal_obs_layer = torch.nn.Linear(input_size, 32)
        self.combination_layer = torch.nn.Linear(32 + 32, 32)
        self.lstm_layer = torch.nn.LSTM(32, 32)
        self.middle_layer = Linear(32, 64)
        self.output_layer = torch.nn.Linear(64, env_spaces.action.n)

    def forward(self, observation, prev_action, prev_reward, rnn_state):
        lead_dim, T, B, _ = infer_leading_dims(observation.state, 1)
        demonstration = observation.demonstration.view(T, B, self.demonstration_length, -1)
        current_obs = observation.state.view(T, B, -1)

        demonstration = relu(self.goal_obs_layer(demonstration[:, :, -1]))
        current_obs = relu(self.current_obs_layer(current_obs))
        x = relu(self.combination_layer(torch.cat((current_obs, demonstration), dim=-1)))

        rnn_state = None if rnn_state is None else tuple(rnn_state)
        lstm_out, (hidden_state, cell_state) = self.lstm_layer(x, rnn_state)
        x = relu(self.middle_layer(lstm_out.view(T * B, -1)))
        q = self.output_layer(x)

        # Restore leading dimensions: [T,B], [B], or [], as input.
        output = restore_leading_dims(q, lead_dim, T, B)
        # Model should always leave B-dimension in rnn state: [N,B,H].
        rnn_state = RnnState(h=hidden_state, c=cell_state)
        return output, rnn_state




class FakeRecurrentDemonstrationQModel(torch.nn.Module):
    def __init__(self, input_size, output_size, demonstration_length=150):
        super().__init__()
        size = 4
        self.demonstration_length = demonstration_length
        self.mlp = MlpModel(input_size + self.demonstration_length * input_size,
                            hidden_sizes=[size * input_size, size * input_size])
        self.output_layer = Linear(size * input_size, output_size)

    def forward(self, observation, prev_action, prev_reward, rnn_state):
        lead_dim, T, B, _ = infer_leading_dims(observation.state, 1)
        current_obs = observation.state.view(T * B, -1)
        demonstration = observation.demonstration.view(T * B, -1)
        output = self.mlp(torch.cat((current_obs, demonstration), dim=-1))
        output = self.output_layer(output)
        output = restore_leading_dims(output, lead_dim, T, B)
        # Model should always leave B-dimension in rnn state: [N,B,H].
        rnn_state = RnnState(h=torch.zeros(1, B, 1), c=torch.zeros(1, B, 1))
        return output, rnn_state


class RecurrentDemonstrationRecurrentQModel(torch.nn.Module):
    def __init__(self, env_spaces):
        super().__init__()
        input_size = env_spaces.observation.shape.state[0]
        action_dim = env_spaces.action.n
        self.demonstration_length = env_spaces.observation.shape.demonstration[0]
        self.lstm_size = 4 * input_size
        self.demonstration_LSTM = LSTM(input_size, hidden_size=self.lstm_size)
        self.combination_layer = torch.nn.Linear(input_size + self.lstm_size, 2 * self.lstm_size)
        self.lstm_layer = torch.nn.LSTM(2 * self.lstm_size, self.lstm_size)
        self.middle_layer = Linear(self.lstm_size, self.lstm_size)
        self.output_layer = torch.nn.Linear(self.lstm_size, action_dim)

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
        x = relu(self.middle_layer(lstm_out))
        output = self.output_layer(x).view(T * B, -1)

        # Restore leading dimensions: [T,B], [B], or [], as input.
        output = restore_leading_dims(output, lead_dim, T, B)
        # Model should always leave B-dimension in rnn state: [N,B,H].
        rnn_state = RnnState(h=hidden_state, c=cell_state)
        return output, rnn_state


class SingleRecurrentDemonstrationRecurrentQModel(torch.nn.Module):
    def __init__(self, env_spaces):
        super().__init__()
        input_size = env_spaces.observation.shape.state[0]
        action_dim = env_spaces.action.n
        self.demonstration_length = env_spaces.observation.shape.demonstration[0]
        self.lstm_size = 4 * input_size
        self.lstm_layer = torch.nn.LSTM(input_size, self.lstm_size)
        self.middle_layer = Linear(self.lstm_size, self.lstm_size)
        self.output_layer = torch.nn.Linear(self.lstm_size, action_dim)

    def forward(self, observation, prev_action, prev_reward, rnn_state):
        lead_dim, T, B, _ = infer_leading_dims(observation.state, 1)
        current_obs = observation.state.view(T, B, -1)
        if rnn_state is None:
            demonstration = observation.demonstration.view(T, B, self.demonstration_length, -1)[0]
            demonstration = demonstration.permute(1, 0, 2)  # now demonstration time steps x Batch x features
            lstm_out, rnn_state = self.lstm_layer(demonstration, None)
            demonstration_encoding = lstm_out[-1]  # now in shape batch x features
        else:
            rnn_state = tuple(rnn_state)

        lstm_out, (hidden_state, cell_state) = self.lstm_layer(current_obs, rnn_state)
        # x = relu(self.combination_layer(torch.cat((current_obs, demonstration_encoding.expand(T, B, -1)), dim=-1)))
        # lstm_out, (hidden_state, cell_state) = self.lstm_layer(x, rnn_state)
        x = relu(self.middle_layer(lstm_out))
        output = self.output_layer(x).view(T * B, -1)

        # Restore leading dimensions: [T,B], [B], or [], as input.
        output = restore_leading_dims(output, lead_dim, T, B)
        # Model should always leave B-dimension in rnn state: [N,B,H].
        rnn_state = RnnState(h=hidden_state, c=cell_state)
        return output, rnn_state


class RecurrentDemonstrationQModel(torch.nn.Module):
    def __init__(self, env_spaces):
        super().__init__()
        input_size = env_spaces.observation.shape.state[0]
        action_dim = env_spaces.action.n
        self.lstm_size = input_size * 4
        self.demonstration_length = env_spaces.observation.shape.demonstration[0]
        self.mlp = MlpModel(input_size + self.demonstration_length * input_size,
                            hidden_sizes=[self.lstm_size, self.lstm_size])
        self.lstm_layer = torch.nn.LSTM(self.lstm_size, self.lstm_size)
        self.middle_layer = Linear(self.lstm_size, self.lstm_size)
        self.output_layer = Linear(self.lstm_size, action_dim)

    def forward(self, observation, prev_action, prev_reward, rnn_state):
        lead_dim, T, B, _ = infer_leading_dims(observation.state, 1)
        current_obs = observation.state.view(T, B, -1)
        demonstration = observation.demonstration.view(T, B, -1)
        output = self.mlp(torch.cat((current_obs, demonstration), dim=-1))

        rnn_state = None if rnn_state is None else tuple(rnn_state)
        # import pdb; pdb.set_trace()
        lstm_out, (hidden_state, cell_state) = self.lstm_layer(output, rnn_state)
        x = relu(self.middle_layer(lstm_out.view(T * B, -1)))
        output = self.output_layer(x)

        output = restore_leading_dims(output, lead_dim, T, B)
        # Model should always leave B-dimension in rnn state: [N,B,H].
        rnn_state = RnnState(h=hidden_state, c=cell_state)
        # rnn_state = RnnState(h=torch.zeros(1, B, 1), c=torch.zeros(1, B, 1))
        return output, rnn_state
