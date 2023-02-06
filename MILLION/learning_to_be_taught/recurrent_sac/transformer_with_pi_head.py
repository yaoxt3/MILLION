
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
import time
import torch
import torch.nn as nn
import numpy as np
from rlpyt.utils.collections import namedarraytuple
from rlpyt.models.mlp import MlpModel
from learning_to_be_taught.models.transformer_models import DemonstrationTransformerModel

RnnState = namedarraytuple("RnnState", ["observations", "length"])


def generate_square_subsequent_mask(size, device='cpu'):
    mask = (torch.triu(torch.ones((size, size), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class TransformerWithPiHead(nn.Module):

    def __init__(self, observation_shape, action_size,
                 num_features=256,
                 demonstration_length=150,
                 **kwargs):
        super().__init__()
        self.input_size = np.prod(observation_shape.state)
        self.action_size = action_size
        self.transformer = DemonstrationTransformerModel(
                input_size=self.input_size,
                output_size=num_features,
                demonstration_length=demonstration_length,
                **kwargs)

        self.shared_model  = MlpModel(self.input_size, [256], num_features)
        # num_features = self.input_size
        self.pi_head = MlpModel(num_features, [256, 256], 2 * action_size)

    def forward(self, observation, prev_action, prev_reward, rnn_state):
        lead_dim, T, B, _ = infer_leading_dims(observation.state, 1)
        features, rnn_state = self.base_model_forward(observation, prev_action, prev_reward, rnn_state)
        mu, log_std = self.pi_head_forward(features)
        return features, mu, log_std, rnn_state


    def pi_head_forward(self, features):
        lead_dim, T, B, _ = infer_leading_dims(features, 1)
        action_dist_params = self.pi_head(features.view(T * B, -1))
        mu, log_std = action_dist_params[:, :self.action_size], action_dist_params[:, self.action_size:]
        mu, log_std = restore_leading_dims((mu, log_std), lead_dim, T, B)
        return mu, log_std

    def base_model_forward(self, observation, prev_action, prev_reward, rnn_state):
        lead_dim, T, B, _ = infer_leading_dims(observation.state, 1)
        features, rnn_state = self.transformer(observation, prev_action, prev_reward, rnn_state)
        # features = self.shared_model(observation.state)
        # rnn_state = torch.zeros(1, B, 1)
        return features, rnn_state

    
    def base_model_parameters(self):
        # return self.shared_model.parameters()
        return self.transformer.parameters()

    def pi_head_parameters(self):
        return self.pi_head.parameters()
