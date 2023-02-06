from rlpyt.models.mlp import MlpModel
import math
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
import torch
from torch import relu
from torch.nn import Linear, LSTMCell, LSTM
from rlpyt.utils.collections import namedarraytuple
from rlpyt.models.mlp import MlpModel

RnnState = namedarraytuple("RnnState", ["h", "c"])

class AttentionModel(torch.nn.Module):
    def __init__(self, input_size, lstm_size, max_demonstration_length):
        super().__init__()
        self.lstm_size = lstm_size
        self.max_demonstration_length = max_demonstration_length
        self.demonstration_encoder = torch.nn.LSTM(input_size, self.lstm_size)
        self.current_obs_layer = torch.nn.Linear(input_size, self.lstm_size)
        self.goal_obs_layer = torch.nn.Linear(input_size, self.lstm_size)
        self.attn = Linear(self.lstm_size, self.lstm_size)
        self.matrix = torch.nn.Parameter(torch.Tensor(self.lstm_size, self.lstm_size))
        self.attn_combine = MlpModel(input_size=input_size + self.lstm_size,
                                     hidden_sizes=[4 * lstm_size,],
                                     output_size=self.lstm_size)
        self.lstm_layer = torch.nn.LSTM(self.lstm_size, self.lstm_size)

    def forward(self, demonstration, demonstration_lengths, current_obs, rnn_state):
        """

        :param demonstration: shape is T x B x time_steps x features
        :param current_obs: shape is T x B x features
        :param rnn_state: None or last rnn_state
        :return:
        """
        lead_dim, T, B, _ = infer_leading_dims(current_obs, 1)
        if rnn_state is None:
            hidden_state, cell_state = (torch.zeros(1, B, self.lstm_size, device=current_obs.device),
                    torch.zeros(1, B, self.lstm_size).to(current_obs.device))
        else:
            (hidden_state, cell_state) = tuple(rnn_state)

        demonstration_encoding, _ = self.demonstration_encoder(demonstration[0].permute(1, 0, 2))
        # change shape to (B * T, demonstration_time_steps, features)
        demonstration_encoding = demonstration_encoding.permute(1, 0, 2)

        q = torch.zeros(T, B, self.lstm_size, device=demonstration.device)
        for t in range(T):
            # (1, B, features) -> (B, 1, features) -> (B, (repeated x demonstration length), features)
            hidden_state_reshape = hidden_state.squeeze(0).unsqueeze(-1)
            attn_weights = torch.bmm(self.attn(demonstration_encoding), hidden_state_reshape).squeeze(-1)
            attn_weights = attn_weights / math.sqrt(self.max_demonstration_length)

            mask = torch.arange(self.max_demonstration_length, device=demonstration.device)[None,
                   :] < demonstration_lengths[:, None]
            mask[:, 0] = True # prevent nans
            attn_weights[~mask] = float('-inf')
            attn_weights = torch.softmax(attn_weights.unsqueeze(1), dim=-1)
            attn_applied = torch.bmm(attn_weights, demonstration_encoding).squeeze(1)

            x = relu(self.attn_combine(torch.cat((attn_applied, current_obs[t]), -1)))
            x, (hidden_state, cell_state) = self.lstm_layer(x.unsqueeze(0), (hidden_state, cell_state))
            q[t] = x

        # Restore leading dimensions: [T,B], [B], or [], as input.
        output = restore_leading_dims(q.reshape(T * B, -1), lead_dim, T, B)
        # Model should always leave B-dimension in rnn state: [N,B,H].
        rnn_state = RnnState(h=hidden_state, c=cell_state)
        return output, rnn_state

class DemonstrationAttentionQModelBidirectional(torch.nn.Module):
    def __init__(self, env_spaces):
        super().__init__()
        self.max_demonstration_length = env_spaces.observation.shape.demonstration[0]
        input_size = env_spaces.observation.shape.state[0]
        self.lstm_size = input_size * 4
        self.action_size = env_spaces.action.n
        self.demonstration_encoder_forward = torch.nn.LSTM(input_size, self.lstm_size)
        self.demonstration_encoder_backward = torch.nn.LSTM(input_size, self.lstm_size)

        self.current_obs_layer = torch.nn.Linear(input_size, self.lstm_size)
        self.goal_obs_layer = torch.nn.Linear(input_size, self.lstm_size)
        self.attn = Linear(3 * self.lstm_size, 1)
        self.attn_combine = Linear(input_size + 2 * self.lstm_size, self.lstm_size)
        self.lstm_layer = torch.nn.LSTM(self.lstm_size, self.lstm_size)
        self.middle_layer = Linear(self.lstm_size, self.lstm_size)
        self.output_layer = torch.nn.Linear(self.lstm_size, env_spaces.action.n)

    def forward(self, observation, prev_action, prev_reward, rnn_state):
        lead_dim, T, B, _ = infer_leading_dims(observation.state, 1)
        demonstration_lengths = observation.demonstration_length.view(T, B)[0]
        current_obs = observation.state.view(T, B, -1)
        # use demonstration provided at first time step
        demonstration = observation.demonstration.view(T, B, self.max_demonstration_length, -1)[0]
        demonstration = demonstration.permute(1, 0, 2)  # shape to (demonstration steps, batch, feature)

        if rnn_state is None:
            hidden_state, cell_state = (torch.zeros(1, B, self.lstm_size), torch.zeros(1, B, self.lstm_size))
        else:
            (hidden_state, cell_state) = tuple(rnn_state)

        forward_encoding, _ = self.demonstration_encoder_forward(demonstration)
        backward_encoding, _ = self.demonstration_encoder_backward(demonstration.flip(dims=(0,)))
        concatenated = torch.cat((forward_encoding, backward_encoding), dim=-1)

        # change shape to (B * T, demonstration_time_steps, features)
        demonstration_encoding = concatenated.permute(1, 0, 2)

        q = torch.zeros(T, B, self.action_size)
        for t in range(T):
            # (1, B, features) -> (B, 1, features) -> (B, (repeated x demonstration length), features)
            hidden_state_reshape = hidden_state.permute(1, 0, 2).repeat((1, self.max_demonstration_length, 1))
            attn_input = torch.cat((demonstration_encoding, hidden_state_reshape), dim=-1)

            attn_weights = self.attn(attn_input).permute(0, 2, 1)
            mask = torch.arange(self.max_demonstration_length, device=demonstration.device)[None,
                   :] < demonstration_lengths[:, None]
            attn_weights[~(mask.unsqueeze(1))] = float('-inf')
            attn_weights = torch.softmax(attn_weights, dim=-1)
            attn_applied = torch.bmm(attn_weights, demonstration_encoding).squeeze(1)

            x = relu(self.attn_combine(torch.cat((attn_applied, current_obs[t]), -1)))
            x, (hidden_state, cell_state) = self.lstm_layer(x.unsqueeze(0), (hidden_state, cell_state))
            x = self.middle_layer(x[0])
            q[t] = self.output_layer(x)

        # Restore leading dimensions: [T,B], [B], or [], as input.
        output = restore_leading_dims(q.reshape(T * B, -1), lead_dim, T, B)
        # Model should always leave B-dimension in rnn state: [N,B,H].
        rnn_state = RnnState(h=hidden_state, c=cell_state)
        return output, rnn_state


class DemonstrationAttentionQModel(torch.nn.Module):
    def __init__(self, env_spaces):
        super().__init__()
        self.max_demonstration_length = env_spaces.observation.shape.demonstration[0]
        input_size = env_spaces.observation.shape.state[0]
        self.lstm_size = input_size * 4
        self.action_size = env_spaces.action.n
        self.demonstration_encoder = torch.nn.LSTM(input_size, self.lstm_size)

        self.current_obs_layer = torch.nn.Linear(input_size, self.lstm_size)
        self.goal_obs_layer = torch.nn.Linear(input_size, self.lstm_size)
        self.attn = Linear(2 * self.lstm_size, 1)
        self.attn_combine = Linear(input_size + self.lstm_size, self.lstm_size)
        self.lstm_layer = torch.nn.LSTM(self.lstm_size, self.lstm_size)
        self.middle_layer = Linear(self.lstm_size, self.lstm_size)
        self.output_layer = torch.nn.Linear(self.lstm_size, env_spaces.action.n)

    def forward(self, observation, prev_action, prev_reward, rnn_state):
        lead_dim, T, B, _ = infer_leading_dims(observation.state, 1)
        demonstration_lengths = observation.demonstration_length.view(T, B)[0]
        demonstration = observation.demonstration.view(T, B, self.max_demonstration_length, -1)
        current_obs = observation.state.view(T, B, -1)

        if rnn_state is None:
            hidden_state, cell_state = (torch.zeros(1, B, self.lstm_size), torch.zeros(1, B, self.lstm_size))
        else:
            (hidden_state, cell_state) = tuple(rnn_state)

        demonstration_encoding, _ = self.demonstration_encoder(demonstration[0].permute(1, 0, 2))
        demonstration_encoding = demonstration_encoding.permute(1, 0,
                                                                2)  # now in (B * T, demonstration_time_steps, features)

        q = torch.zeros(T, B, self.action_size)
        for t in range(T):
            # (1, B, features) -> (B, 1, features) -> (B, (repeated x demonstration length), features)
            hidden_state_reshape = hidden_state.permute(1, 0, 2).repeat((1, self.max_demonstration_length, 1))
            attn_input = torch.cat((demonstration_encoding, hidden_state_reshape), dim=-1)

            attn_weights = self.attn(attn_input).permute(0, 2, 1)
            mask = torch.arange(self.max_demonstration_length, device=demonstration.device)[None,
                   :] < demonstration_lengths[:, None]
            attn_weights[~(mask.unsqueeze(1))] = float('-inf')
            attn_weights = torch.softmax(attn_weights, dim=-1)
            attn_applied = torch.bmm(attn_weights, demonstration_encoding).squeeze(1)

            x = relu(self.attn_combine(torch.cat((attn_applied, current_obs[t]), -1)))
            x, (hidden_state, cell_state) = self.lstm_layer(x.unsqueeze(0), (hidden_state, cell_state))
            x = self.middle_layer(x[0])
            q[t] = self.output_layer(x)

        # Restore leading dimensions: [T,B], [B], or [], as input.
        output = restore_leading_dims(q.reshape(T * B, -1), lead_dim, T, B)
        # Model should always leave B-dimension in rnn state: [N,B,H].
        rnn_state = RnnState(h=hidden_state, c=cell_state)
        return output, rnn_state


class DemonstrationAttentionDotProductQModel(torch.nn.Module):
    def __init__(self, env_spaces):
        super().__init__()
        self.max_demonstration_length = env_spaces.observation.shape.demonstration[0]
        input_size = env_spaces.observation.shape.state[0]
        self.lstm_size = input_size * 4
        self.action_size = env_spaces.action.n
        self.demonstration_encoder = torch.nn.LSTM(input_size, self.lstm_size)

        self.current_obs_layer = torch.nn.Linear(input_size, self.lstm_size)
        self.goal_obs_layer = torch.nn.Linear(input_size, self.lstm_size)
        self.attn = Linear(self.lstm_size, self.lstm_size)
        self.matrix = torch.nn.Parameter(torch.Tensor(self.lstm_size, self.lstm_size))
        self.attn_combine = Linear(input_size + self.lstm_size, self.lstm_size)
        self.lstm_layer = torch.nn.LSTM(self.lstm_size, self.lstm_size)
        self.middle_layer = Linear(self.lstm_size, self.lstm_size)
        self.output_layer = torch.nn.Linear(self.lstm_size, env_spaces.action.n)

    def forward(self, observation, prev_action, prev_reward, rnn_state):
        lead_dim, T, B, _ = infer_leading_dims(observation.state, 1)
        demonstration_lengths = observation.demonstration_length.view(T, B)[0]
        demonstration = observation.demonstration.view(T, B, self.max_demonstration_length, -1)
        current_obs = observation.state.view(T, B, -1)

        if rnn_state is None:
            hidden_state, cell_state = (torch.zeros(1, B, self.lstm_size), torch.zeros(1, B, self.lstm_size))
        else:
            (hidden_state, cell_state) = tuple(rnn_state)

        demonstration_encoding, _ = self.demonstration_encoder(demonstration[0].permute(1, 0, 2))
        # change shape to (B * T, demonstration_time_steps, features)
        demonstration_encoding = demonstration_encoding.permute(1, 0, 2)

        q = torch.zeros(T, B, self.action_size)
        for t in range(T):
            # (1, B, features) -> (B, 1, features) -> (B, (repeated x demonstration length), features)
            hidden_state_reshape = hidden_state.squeeze(0).unsqueeze(-1)
            attn_weights = torch.bmm(self.attn(demonstration_encoding), hidden_state_reshape).squeeze(-1)
            attn_weights = attn_weights / math.sqrt(self.max_demonstration_length)

            mask = torch.arange(self.max_demonstration_length, device=demonstration.device)[None,
                   :] < demonstration_lengths[:, None]
            attn_weights[~mask] = float('-inf')
            attn_weights = torch.softmax(attn_weights.unsqueeze(1), dim=-1)
            print('attention weights: '+ str(attn_weights))
            attn_applied = torch.bmm(attn_weights, demonstration_encoding).squeeze(1)

            x = relu(self.attn_combine(torch.cat((attn_applied, current_obs[t]), -1)))
            x, (hidden_state, cell_state) = self.lstm_layer(x.unsqueeze(0), (hidden_state, cell_state))
            x = self.middle_layer(x[0])
            q[t] = self.output_layer(x)

        # Restore leading dimensions: [T,B], [B], or [], as input.
        output = restore_leading_dims(q.reshape(T * B, -1), lead_dim, T, B)
        # Model should always leave B-dimension in rnn state: [N,B,H].
        rnn_state = RnnState(h=hidden_state, c=cell_state)
        return output, rnn_state


class FixedLengthDemonstrationAttentionQModel(torch.nn.Module):
    def __init__(self, env_spaces):
        super().__init__()
        self.demonstration_length = env_spaces.observation.shape.demonstration[0]
        input_size = env_spaces.observation.shape.state[0]
        self.lstm_size = 32
        self.action_size = env_spaces.action.n

        self.current_obs_layer = torch.nn.Linear(input_size, 32)
        self.goal_obs_layer = torch.nn.Linear(input_size, 32)
        self.attn = Linear(self.lstm_size + input_size, self.demonstration_length)
        self.attn_combine = Linear(2 * input_size, self.lstm_size)
        self.lstm_layer = torch.nn.LSTM(self.lstm_size, self.lstm_size)
        self.middle_layer = Linear(self.lstm_size, 2 * self.lstm_size)
        self.output_layer = torch.nn.Linear(2 * self.lstm_size, env_spaces.action.n)

    def forward(self, observation, prev_action, prev_reward, rnn_state):
        lead_dim, T, B, _ = infer_leading_dims(observation.state, 1)
        demonstration = observation.demonstration.view(T, B, self.demonstration_length, -1)
        current_obs = observation.state.view(T, B, -1)

        if rnn_state is None:
            hidden_state, cell_state = (torch.zeros(1, B, self.lstm_size), torch.zeros(1, B, self.lstm_size))
        else:
            (hidden_state, cell_state) = tuple(rnn_state)

        q = torch.zeros(T, B, self.action_size)
        for t in range(T):
            attn_weights = torch.softmax(self.attn(torch.cat((current_obs[t], hidden_state[0]), -1)), dim=1)
            attn_applied = torch.bmm(attn_weights.unsqueeze(1), demonstration[t]).squeeze(1)

            x = relu(self.attn_combine(torch.cat((attn_applied, current_obs[t]), -1)))
            x, (hidden_state, cell_state) = self.lstm_layer(x.unsqueeze(0), (hidden_state, cell_state))
            x = self.middle_layer(x[0])
            q[t] = self.output_layer(x)

        # Restore leading dimensions: [T,B], [B], or [], as input.
        output = restore_leading_dims(q.reshape(T * B, -1), lead_dim, T, B)
        # Model should always leave B-dimension in rnn state: [N,B,H].
        rnn_state = RnnState(h=hidden_state, c=cell_state)
        return output, rnn_state
