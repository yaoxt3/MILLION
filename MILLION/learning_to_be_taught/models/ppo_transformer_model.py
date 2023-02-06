import numpy as np
import torch

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.models.mlp import MlpModel
from rlpyt.models.running_mean_std import RunningMeanStdModel
from rlpyt.utils.collections import namedarraytuple
from rlpyt.models.mlp import MlpModel
from learning_to_be_taught.models.transformer_models import DemonstrationTransformerModel

RnnState = namedarraytuple("RnnState", ["observations", "length"])


class PpoTransformerModel(torch.nn.Module):
    """
    Recurrent model for Mujoco locomotion agents: an MLP into an LSTM which
    outputs distribution means, log_std, and state-value estimate.
    """

    def __init__(
            self,
            observation_shape,
            action_size,
            hidden_sizes=None,  # None for default (see below).
            max_demonstration_length=150,
            **kwargs
    ):
        super().__init__()
        self._obs_n_dim = len(observation_shape)
        self.action_size = action_size
        self.input_size = np.prod(observation_shape.state)
        num_features = 256
        self.transformer = DemonstrationTransformerModel(
            input_size=self.input_size,
            output_size=num_features,
            demonstration_length=max_demonstration_length,
            **kwargs)

        self.value_head = MlpModel(num_features, [256, 256], 1)
        self.pi_head = MlpModel(num_features, [256, 256], 2 * action_size)

    def forward(self, observation, prev_action, prev_reward, init_rnn_state):
        """
        Compute mean, log_std, and value estimate from input state. Infer
        leading dimensions of input: can be [T,B], [B], or []; provides
        returns with same leading dims.  Intermediate feedforward layers
        process as [T*B,H], and recurrent layers as [T,B,H], with T=1,B=1 when
        not given. Used both in sampler and in algorithm (both via the agent).
        Also returns the next RNN state.
        """
        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, _ = infer_leading_dims(observation.state, 1)
        features, rnn_state = self.transformer(observation, prev_action, prev_reward, init_rnn_state)
        lead_dim, T, B, _ = infer_leading_dims(features, 1)
        action_dist_params = self.pi_head(features.view(T * B, -1))
        mu, log_std = action_dist_params[:, :self.action_size], action_dist_params[:, self.action_size:]
        # mu, log_std = restore_leading_dims((mu, log_std), lead_dim, T, B)

        value = self.value_head(features.view(T * B, -1)).squeeze(-1)
        mu, log_std, value = restore_leading_dims((mu, log_std, value), lead_dim, T, B)
        return mu, log_std, value, rnn_state

    def update_obs_rms(self, observation):
        pass
