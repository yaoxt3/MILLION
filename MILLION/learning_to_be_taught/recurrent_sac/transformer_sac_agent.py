import torch
from rlpyt.agents.base import AgentStep
from learning_to_be_taught.recurrent_sac.sac_attention_model import PiAttentionModel, QAttentionModel
from rlpyt.utils.buffer import buffer_func, buffer_method
from rlpyt.utils.buffer import buffer_to
from rlpyt.utils.collections import namedarraytuple
from rlpyt.agents.qpg.sac_agent import SacAgent
from rlpyt.agents.base import RecurrentAgentMixin
from rlpyt.distributions.gaussian import DistInfoStd
from rlpyt.agents.base import AgentInputs

AgentInfo = namedarraytuple("AgentInfo", ["dist_info", "prev_rnn_state"])
# AgentInputs = namedarraytuple("ModelInput", ["observation", "prev_action", 'prev_reward'])
RnnState = namedarraytuple('RnnState', ['pi', 'q1', 'q2'])
QValues = namedarraytuple('QValues', ['q1', 'q2', 'target_q1', 'target_q2'])
QRnnState = namedarraytuple('QRnnState', ['q1', 'q2', 'target_q1', 'target_q2'])

MIN_LOG_STD = -20
MAX_LOG_STD = 2


class TransformerSacAgent(RecurrentAgentMixin, SacAgent):
    """Base agent for recurrent DQN (to add recurrent mixin)."""

    def __init__(self,
                 ModelCls=PiAttentionModel,
                 QModelCls=QAttentionModel,
                 **kwargs):
        """
        Arguments are saved but no model initialization occurs.

        Args:
            ModelCls: The model class to be used.
            model_kwargs (optional): Any keyword arguments to pass when instantiating the model.
            initial_model_state_dict (optional): Initial model parameter values.
        """
        super().__init__(ModelCls=ModelCls, QModelCls=QModelCls, **kwargs)
        self.model_kwargs = dict()
        self.q_model_kwargs = dict()


    def all_q(self, model_input: AgentInputs, init_rnn_states: QRnnState, action=None) -> (QValues, QRnnState):
        """Compute twin Q-values for state/observation and input action
        (with grad)."""
        model_input = buffer_to(model_input, device=self.device)
        q1_rnn_state, q2_rnn_state = buffer_to((init_rnn_states.q1, init_rnn_states.q2), self.device)
        target_q1_rnn_state, target_q2_rnn_state = buffer_to((init_rnn_states.target_q1, init_rnn_states.target_q2),
                                                             self.device)
        action = buffer_to(action, device=self.device)
        q1, q1_rnn_state = self.q1_model(*model_input, action, q1_rnn_state)
        q2, q2_rnn_state = self.q2_model(*model_input, action, q2_rnn_state)
        with torch.no_grad():
            target_q1, target_q1_rnn_state = self.target_q1_model(*model_input, action, target_q1_rnn_state)
            target_q2, target_q2_rnn_state = self.target_q2_model(*model_input, action, target_q2_rnn_state)
        q_values = QValues(q1=q1, q2=q2, target_q1=target_q1, target_q2=target_q2)
        rnn_states = QRnnState(q1=q1_rnn_state, q2=q2_rnn_state, target_q1=target_q1_rnn_state,
                               target_q2=target_q2_rnn_state)
        q_values = buffer_to(q_values, device='cpu')
        return q_values, rnn_states

    def q(self, model_input, q1_rnn_state, q2_rnn_state, action=None):
        model_input = buffer_to(model_input, device=self.device)
        q1_rnn_state, q2_rnn_state = buffer_to((q1_rnn_state, q2_rnn_state), self.device)
        action = buffer_to(action, device=self.device)
        q1, q1_rnn_state = self.q1_model(*model_input, action, q1_rnn_state)
        q2, q2_rnn_state = self.q2_model(*model_input, action, q2_rnn_state)
        return q1, q2

    def target_q(self, model_input, q1_rnn_state, q2_rnn_state, action=None):
        model_input = buffer_to(model_input, device=self.device)
        q1_rnn_state, q2_rnn_state = buffer_to((q1_rnn_state, q2_rnn_state), self.device)
        action = buffer_to(action, device=self.device)
        q1, q1_rnn_state = self.target_q1_model(*model_input, action, q1_rnn_state)
        q2, q2_rnn_state = self.target_q2_model(*model_input, action, q2_rnn_state)
        return q1, q2

    def pi(self, model_input, init_rnn_state):
        """Compute action log-probabilities for state/observation, and
        sample new action (with grad).  Uses special ``sample_loglikelihood()``
        method of Gaussian distriution, which handles action squashing
        through this process."""
        model_input = buffer_to(model_input, device=self.device)
        init_rnn_state = buffer_to(init_rnn_state, self.device)
        mean, log_std, rnn_state = self.model(*model_input, init_rnn_state)
        dist_info = DistInfoStd(mean=mean, log_std=log_std)
        action, log_pi = self.distribution.sample_loglikelihood(dist_info)
        # log_pi, dist_info = buffer_to((log_pi, dist_info), device="cpu")
        return action, log_pi, dist_info, rnn_state  # Action stays on device for q models.

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        model_inputs = buffer_to((observation, prev_action, prev_reward), device=self.device)
        pi_rnn_state, q1_rnn_state, q2_rnn_state = self.decode_rnn_state()
        mean, log_std, pi_rnn_state = self.model(*model_inputs, pi_rnn_state)
        _, q1_rnn_state = self.q1_model(*model_inputs, torch.zeros_like(mean), q1_rnn_state)
        _, q2_rnn_state = self.q2_model(*model_inputs, torch.zeros_like(mean), q2_rnn_state)

        dist_info = DistInfoStd(mean=mean, log_std=log_std)
        action = self.distribution.sample(dist_info)

        new_rnn_state = RnnState(pi=pi_rnn_state, q1=q1_rnn_state, q2=q2_rnn_state)
        prev_rnn_state = self.prev_rnn_state or buffer_func(new_rnn_state, torch.zeros_like)
        # Transpose the rnn_state from [N,B,H] --> [B,N,H] for storage.
        # (Special case, model should always leave B dimension in.)
        prev_rnn_state = buffer_method(prev_rnn_state, "transpose", 0, 1)
        prev_rnn_state = buffer_to(prev_rnn_state, device="cpu")
        agent_info = AgentInfo(dist_info=dist_info, prev_rnn_state=prev_rnn_state)

        action, agent_info = buffer_to((action, agent_info), device="cpu")
        self.advance_rnn_state(new_rnn_state)
        return AgentStep(action=action, agent_info=agent_info)

    def decode_rnn_state(self):
        if self.prev_rnn_state is None:
            return None, None, None
        else:
            return self.prev_rnn_state.pi, self.prev_rnn_state.q1, self.prev_rnn_state.q2

    def make_env_to_model_kwargs(self, env_spaces):
        assert len(env_spaces.action.shape) == 1
        return dict(
            observation_shape=env_spaces.observation.shape,
            action_size=env_spaces.action.shape[0],
        )
