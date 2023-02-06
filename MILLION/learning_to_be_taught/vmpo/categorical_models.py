import numpy as np
import torch

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.models.conv2d import Conv2dModel
from rlpyt.models.mlp import MlpModel
from rlpyt.models.dqn.dueling import DuelingHeadModel

class CategorialFfModel(torch.nn.Module):
    """Standard convolutional network for DQN.  2-D convolution for multiple
    video frames per observation, feeding an MLP for Q-value outputs for
    the action set.
    """

    def __init__(
            self,
            observation_shape,
            action_size,
            linear_value_output=True,
            seperate_value_network=False
    ):
        """Instantiates the neural network according to arguments; network defaults
        stored within this method."""
        super().__init__()
        self.shared_dim = 256
        self.shared_mlp = MlpModel(
            input_size=np.prod(observation_shape),
            hidden_sizes=[256, self.shared_dim],
        )
        if seperate_value_network:
            self.seperate_value_network = MlpModel(
                input_size=np.prod(observation_shape),
                hidden_sizes=[256, self.shared_dim],
            )
        self.pi_head = MlpModel(self.shared_dim, [256,], action_size)
        self.value_head = MlpModel(self.shared_dim, [256,], 1 if linear_value_output else None)

    def forward(self, observation, prev_action, prev_reward, init_rnn_state):
        # obs = observation.type(torch.float)  # Expect torch.uint8 inputs
        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, img_shape = infer_leading_dims(observation, 1)
        obs = observation.reshape(T * B, -1)

        features = self.shared_mlp(obs)
        pi = self.pi_head(features)
        pi = torch.softmax(pi, dim=-1)

        value_features = self.seperate_value_network(obs) if self.seperate_value_network else features
        value = self.value_head(value_features).squeeze(-1)
        # Restore leading dimensions: [T,B], [B], or [], as input.
        pi, value = restore_leading_dims((pi, value), lead_dim, T, B)
        return pi, value, torch.zeros(1, B, 1)
