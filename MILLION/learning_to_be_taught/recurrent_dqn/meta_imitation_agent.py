import torch
from rlpyt.agents.base import (AgentStep)
from rlpyt.utils.logging import logger
from rlpyt.agents.dqn.dqn_agent import DqnAgent
from rlpyt.utils.buffer import buffer_to, buffer_func, buffer_method
from rlpyt.utils.collections import namedarraytuple
from learning_to_be_taught.models.recurrent_models import DemonstrationRecurrentQModel
from rlpyt.agents.base import RecurrentAgentMixin

AgentInfo = namedarraytuple("AgentInfo", ["q", "prev_rnn_state"])


class MetaImitationAgent(RecurrentAgentMixin, DqnAgent):
    """Base agent for recurrent DQN (to add recurrent mixin)."""

    def __init__(self, ModelCls=DemonstrationRecurrentQModel, eps_eval=0, **kwargs):
        """
        Arguments are saved but no model initialization occurs.

        Args:
            ModelCls: The model class to be used.
            model_kwargs (optional): Any keyword arguments to pass when instantiating the model.
            initial_model_state_dict (optional): Initial model parameter values.
        """

        super().__init__(ModelCls=ModelCls, eps_eval=eps_eval, **kwargs)

    def initialize(self, *args, **kwargs):
        _initial_model_state_dict = self.initial_model_state_dict
        self.initial_model_state_dict = None  # don't let base agent try to initialize model
        super().initialize(*args, **kwargs)
        if _initial_model_state_dict is not None:
            self.model.load_state_dict(_initial_model_state_dict['model'])
            self.target_model.load_state_dict(_initial_model_state_dict['model'])

    def __call__(self, observation, prev_action, prev_reward, init_rnn_state):
        # Assume init_rnn_state already shaped: [N,B,H]
        prev_action = self.distribution.to_onehot(prev_action)
        model_inputs = buffer_to((observation, prev_action, prev_reward,
                                  init_rnn_state), device=self.device)
        q, rnn_state = self.model(*model_inputs)
        return q.cpu(), rnn_state  # Leave rnn state on device.

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        """Computes Q-values for states/observations and selects actions by
        epsilon-greedy (no grad).  Advances RNN state."""
        prev_action = self.distribution.to_onehot(prev_action)
        agent_inputs = buffer_to((observation, prev_action, prev_reward),
                                 device=self.device)
        q, rnn_state = self.model(*agent_inputs, self.prev_rnn_state)  # Model handles None.
        q = q.cpu()
        action = self.distribution.sample(q)
        prev_rnn_state = self.prev_rnn_state or buffer_func(rnn_state, torch.zeros_like)
        # Transpose the rnn_state from [N,B,H] --> [B,N,H] for storage.
        # (Special case, model should always leave B dimension in.)
        prev_rnn_state = buffer_method(prev_rnn_state, "transpose", 0, 1)
        prev_rnn_state = buffer_to(prev_rnn_state, device="cpu")
        agent_info = AgentInfo(q=q, prev_rnn_state=prev_rnn_state)
        self.advance_rnn_state(rnn_state)  # Keep on device.
        return AgentStep(action=action, agent_info=agent_info)

    def target(self, observation, prev_action, prev_reward, init_rnn_state):
        # Assume init_rnn_state already shaped: [N,B,H]
        prev_action = self.distribution.to_onehot(prev_action)
        model_inputs = buffer_to((observation, prev_action, prev_reward, init_rnn_state),
                                 device=self.device)
        target_q, rnn_state = self.target_model(*model_inputs)
        return target_q.cpu(), rnn_state  # Leave rnn state on device.

    def make_env_to_model_kwargs(self, env_spaces):
        """Generate any keyword args to the model which depend on environment interfaces."""
        return {'env_spaces': env_spaces}


    def eval_mode(self, itr):
        """Extend method to set epsilon for evaluation, using 1 for
        pre-training eval."""
        super().eval_mode(itr)
        logger.log(f"Meta imitation Agent at itr {itr}, eval eps "
                   f"{self.eps_eval}")
        self.distribution.set_epsilon(self.eps_eval)
