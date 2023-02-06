from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
import time
import torch
import torch.nn as nn
from torch.nn import Transformer
from learning_to_be_taught.models.custom_transformer_layers import CustomTransformer
from learning_to_be_taught.models.gated_transformer import GatedTransformer, SIZES
import numpy as np
from rlpyt.utils.collections import namedarraytuple
from rlpyt.models.mlp import MlpModel

RnnState = namedarraytuple("RnnState", ["observations", "length"])


def generate_square_subsequent_mask(size, device='cpu'):
    mask = (torch.triu(torch.ones((size, size), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class DemonstrationTransformerModel(nn.Module):

    def __init__(self, input_size, output_size, demonstration_length=150,
                 reward_and_action_observation=False,
                 size='medium',
                 **kwargs):
        super().__init__()
        self.input_size = input_size
        self.reward_and_action_observation = reward_and_action_observation
        self.d_model = SIZES[size]['d_model']
        self.output_size = output_size
        self.demonstration_length = demonstration_length
        self.episode_length = demonstration_length - 1

        self.demonstration_encoding_layer = torch.nn.Linear(self.input_size, self.d_model)
        self.observation_size = self.input_size + 2 if reward_and_action_observation else self.input_size
        self.observations_encoding_layer = torch.nn.Linear(self.observation_size, self.d_model)
        self.transformer = GatedTransformer(**SIZES[size], **kwargs)
        self.output_layer = torch.nn.Linear(self.d_model, self.output_size)

        self.subsequent_mask = generate_square_subsequent_mask(self.episode_length)
        self.layer  = MlpModel(input_size, [256,], output_size)

    def forward(self, observation, prev_action, prev_reward, rnn_state):
        lead_dim, T, B, _ = infer_leading_dims(observation.state, 1)
        demonstration = observation.demonstration.view(T, B, -1, self.input_size)[0].permute(1, 0, 2)
        self.subsequent_mask = self.subsequent_mask.to(demonstration.device)

        if T == 1:
            # non differentiable sampling
            if rnn_state is None:
                observations = torch.zeros((self.episode_length, B, self.observation_size), device=observation.state.device)
                length = torch.zeros(1, B, 1)
            else:
                observations = rnn_state.observations
                length = rnn_state.length.clamp_max(self.demonstration_length - 2)

            # write new observations in tensor with older observations
            for b in range(B):
                observations[int(length[0, b]):int(length[0, b]) + T, b] = observation.state.view(T, B, -1)[:, b]
            demonstration = self.demonstration_encoding_layer(demonstration)
            transformer_output = self.transformer(demonstration, self.observations_encoding_layer(observations),
                                                 tgt_mask=self.subsequent_mask)
            transformer_output = self.output_layer(transformer_output)
            # transformer_output = self.layer(observations)

            output = torch.zeros((T, B, self.output_size), device=observation.state.device)
            for b in range(B):
                output[:, b] = transformer_output[int(length[0, b]):int(length[0, b]) + T, b]

            rnn_state = RnnState(observations=observations, length=length + 1)
        else:
            subsequent_mask = generate_square_subsequent_mask(T, device=observation.state.device)
            demonstration = self.demonstration_encoding_layer(demonstration)
            observation = self.observations_encoding_layer(observation.state.view(T, B, -1))
            output = self.transformer(demonstration, observation, tgt_mask=subsequent_mask)
            output = self.output_layer(output).squeeze()
            # output = self.layer(observation.state.view(T, B, -1))
            rnn_state = None

        output = restore_leading_dims(output.view(T * B, -1), lead_dim, T, B)

        return output, rnn_state
