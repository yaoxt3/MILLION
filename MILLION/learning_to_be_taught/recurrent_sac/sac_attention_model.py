import torch
from learning_to_be_taught.models.attention_models import AttentionModel
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.models.mlp import  MlpModel

class PiAttentionModel(torch.nn.Module):
    """Action distrubition MLP model for SAC agent."""

    def __init__(
            self,
            observation_shape,
            action_size,
            hidden_sizes,
            lstm_size=32
    ):
        super().__init__()
        self.max_demonstration_length = observation_shape.demonstration[0]
        self._action_size = action_size
        self.attention_model = AttentionModel(input_size=observation_shape.state[0],
                                              lstm_size=lstm_size,
                                              max_demonstration_length=self.max_demonstration_length)
        self.pi_head = MlpModel(input_size=lstm_size + observation_shape.state[0],
                                hidden_sizes=[2 * lstm_size,],
                                output_size=2 * action_size)

    def forward(self, observation, prev_action, prev_reward, rnn_state):
        lead_dim, T, B, _ = infer_leading_dims(observation.state, 1)
        demonstration_lengths = observation.demonstration_length.view(T, B)[0]
        demonstration = observation.demonstration.view(T, B, self.max_demonstration_length, -1)
        current_obs = observation.state.view(T, B, -1)
        output, rnn_state = self.attention_model(demonstration, demonstration_lengths, current_obs, rnn_state)
        pi_output = self.pi_head(torch.cat((output, current_obs), dim=-1))

        # output = self.mlp(observation.view(T * B, -1))
        # mu, log_std = output[:, :self._action_size], output[:, self._action_size:]
        mu, log_std = pi_output[:, :, :self._action_size], pi_output[:, : , self._action_size:]
        mu, log_std = restore_leading_dims((mu.view(T * B, -1), log_std.view(T * B, -1)), lead_dim, T, B)
        return mu, log_std, rnn_state


class QAttentionModel(torch.nn.Module):
    """Q portion of the model for DDPG, an MLP."""

    def __init__(
            self,
            observation_shape,
            hidden_sizes,
            action_size,
    ):
        """Instantiate neural net according to inputs."""
        super().__init__()
        self.max_demonstration_length = observation_shape.demonstration[0]
        self._action_size = action_size
        lstm_size = 32
        self.attention_model = AttentionModel(input_size=observation_shape.state[0],
                                              lstm_size=lstm_size,
                                              max_demonstration_length=self.max_demonstration_length)
        self.q_head = MlpModel(input_size=lstm_size + action_size +  observation_shape.state[0],
                                hidden_sizes=[2 * lstm_size,],
                                output_size=1)

    def forward(self, observation, prev_action, prev_reward, action, rnn_state):
        lead_dim, T, B, _ = infer_leading_dims(observation.state, 1)
        demonstration_lengths = observation.demonstration_length.view(T, B)[0]
        demonstration = observation.demonstration.view(T, B, self.max_demonstration_length, -1)
        current_obs = observation.state.view(T, B, -1)
        output, rnn_state = self.attention_model(demonstration, demonstration_lengths, current_obs, rnn_state)

        if action is not None:
            q = self.q_head(torch.cat((output.view(T * B, -1), action.view(T * B, -1), current_obs.view(T * B, -1)), dim=-1)).squeeze(-1)
        else:
            q = torch.zeros(T * B)

        q = restore_leading_dims(q, lead_dim, T, B)
        return q, rnn_state
