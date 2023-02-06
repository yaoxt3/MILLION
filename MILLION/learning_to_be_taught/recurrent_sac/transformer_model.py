from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
import time
import torch
import torch.nn as nn
import numpy as np
from rlpyt.utils.collections import namedarraytuple
from rlpyt.models.mlp import MlpModel
from learning_to_be_taught.models.transformer_models import DemonstrationTransformerModel
from learning_to_be_taught.models.transformer_models import generate_square_subsequent_mask
from learning_to_be_taught.models.gated_transformer import GatedTransformer, SIZES
from learning_to_be_taught.models.custom_transformer_layers import CustomTransformer


RnnState = namedarraytuple("RnnState", ["observations", "length"])


class PiTransformerModel(nn.Module):

    def __init__(self, observation_shape, action_size, hidden_sizes=None, size='medium', episode_length=150):
        super().__init__()
        self.state_size = np.prod(observation_shape.state)
        self.episode_length = episode_length
        self.action_size = action_size
        self.d_model = SIZES[size]['d_model']
        self.transformer = GatedTransformer(**SIZES[size], style='reordered')
        self.state_transform = torch.nn.Linear(np.prod(observation_shape.state), self.d_model)
        self.output_layer = MlpModel(self.d_model, [256, 256], 2 * action_size)
        # self.output_layer = torch.nn.Linear(self.d_model, 2 * action_size)
        self.subsequent_mask = generate_square_subsequent_mask(episode_length)

    def forward(self, observation, prev_action, prev_reward, rnn_state):
        lead_dim, T, B, _ = infer_leading_dims(observation.state, 1)
        demonstration = observation.demonstration.view(T, B, -1, self.state_size)[0].permute(1, 0, 2)
        self.subsequent_mask = self.subsequent_mask.to(demonstration.device)
        if T == 1:
            # non differentiable sampling
            if rnn_state is None:
                observations = torch.zeros((self.episode_length, B, self.state_size), device=observation.state.device)
                length = torch.zeros(1, B, 1)
            else:
                observations = rnn_state.observations
                length = rnn_state.length

            # write new observations in tensor with older observations
            for b in range(B):
                observations[int(length[0, b]):int(length[0, b]) + T, b] = observation.state.view(T, B, -1)[:, b]

            source = self.state_transform(demonstration)
            target = self.state_transform(observations)
            transformer_output = self.transformer(source, target, tgt_mask=self.subsequent_mask)
            transformer_output = self.output_layer(transformer_output)
            # transformer_output = self.output_layer(target)

            output = torch.zeros((T, B, 2 * self.action_size), device=observation.state.device)
            for b in range(B):
                output[:, b] = transformer_output[int(length[0, b]):int(length[0, b]) + T, b]

            rnn_state = RnnState(observations=observations, length=length + 1)
        else:
            subsequent_mask = generate_square_subsequent_mask(T, device=observation.state.device)
            source = self.state_transform(demonstration)
            target = self.state_transform(observation.state.view(T, B, -1))
            output = self.transformer(source, target, tgt_mask=subsequent_mask)
            # output = self.output_layer(target).squeeze()
            output = self.output_layer(output).squeeze()
            rnn_state = None

        output = output.view(T * B, -1)
        mu, log_std = output[:, :self.action_size], output[:, self.action_size:]
        mu, log_std = restore_leading_dims((mu, log_std), lead_dim, T, B)

        return mu, log_std, rnn_state


class QTransformerModel(nn.Module):

    def __init__(self, observation_shape, action_size, hidden_sizes=None, size='medium', episode_length=150, state_action_input=True):
        super().__init__()
        self.state_size = np.prod(observation_shape.state)
        self.episode_length = episode_length
        self.state_action_input = state_action_input
        self.action_size = action_size
        self.transformer_input_size = SIZES[size]['d_model']
        self.transformer = GatedTransformer(**SIZES[size], style='reordered')
        self.state_transform = torch.nn.Linear(self.state_size, self.transformer_input_size)
        self.state_action_transform = torch.nn.Linear(self.state_size + action_size, self.transformer_input_size)
        # self.output_layer = torch.nn.Linear(self.transformer_input_size, 1)
        output_layer_input_size = self.transformer_input_size if self.state_action_input else self.transformer_input_size + self.action_size
        self.output_layer = MlpModel(output_layer_input_size, [256, 256], 1)

    def forward(self, observation, prev_action, prev_reward, action, rnn_state):
        lead_dim, T, B, _ = infer_leading_dims(observation.state, 1)
        demonstration = observation.demonstration.view(T, B, -1, self.state_size)[0].permute(1,
                                                                                                              0,
                                                                                                              2)
        if T == 1:
            # return zeros because required by recurrent sac algo but useless for transformer models
            output = torch.zeros(T * B, 1)
            rnn_state = torch.zeros(1, B, 1)
        else:
            observations = observation.state.view(T, B, -1)
            actions = action.view(T, B, -1)
            source = self.state_transform(demonstration)
            if self.state_action_input:
                target = self.state_action_transform(torch.cat((observations, actions), -1))
            else:
                target = self.state_transform(observations)

            subsequent_mask = generate_square_subsequent_mask(T, device=action.device)
            output = self.transformer(source, target, tgt_mask=subsequent_mask)
            if self.state_action_input:
                output = self.output_layer(output).squeeze(-1)
            else:
                output = self.output_layer(torch.cat((output, action), dim=-1)).squeeze(-1)
            rnn_state = None

        output = restore_leading_dims(output.view(T * B), lead_dim, T, B)

        return output, rnn_state
