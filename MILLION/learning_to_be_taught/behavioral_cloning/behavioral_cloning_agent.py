import numpy as np
from rlpyt.utils.buffer import buffer_func, buffer_method
from rlpyt.agents.base import RecurrentAgentMixin
import torch
from collections import namedtuple
from torch.nn.parallel import DistributedDataParallel as DDP
from rlpyt.agents.base import BaseAgent, AgentStep
from rlpyt.utils.quick_args import save__init__args
from rlpyt.distributions.gaussian import Gaussian, DistInfoStd
from rlpyt.utils.buffer import buffer_to
from rlpyt.utils.logging import logger
from rlpyt.models.utils import update_state_dict
from rlpyt.utils.collections import namedarraytuple
from learning_to_be_taught.models.transformer_models import DemonstrationTransformerModel

MIN_LOG_STD = -20
MAX_LOG_STD = 2

AgentInfo = namedarraytuple("AgentInfo", ["action", "prev_rnn_state"])
Models = namedtuple("Models", ["pi", "q1", "q2", "v"])


class BehavioralCloningAgent(RecurrentAgentMixin, BaseAgent):
    """Agent for SAC algorithm, including action-squashing, using twin Q-values."""

    def __init__(
            self,
            ModelCls=DemonstrationTransformerModel,  # Pi model.
            action_squash=1.,  # Max magnitude (or None).
            pretrain_std=0.75,  # With squash 0.75 is near uniform.
            **kwargs
    ):
        """Saves input arguments; network defaults stored within."""
        BaseAgent.__init__(self, ModelCls=ModelCls, **kwargs)
        RecurrentAgentMixin.__init__(self)
        save__init__args(locals())
        self.min_itr_learn = 0  # Get from algo.

    def initialize(self, env_spaces, share_memory=False,
                   global_B=1, env_ranks=None):
        super().initialize(env_spaces, share_memory,
                           global_B=global_B, env_ranks=env_ranks)
        assert len(env_spaces.action.shape) == 1
        self.distribution = Gaussian(
            dim=env_spaces.action.shape[0],
            squash=self.action_squash,
            min_std=np.exp(MIN_LOG_STD),
            max_std=np.exp(MAX_LOG_STD),
        )

    def data_parallel(self):
        device_id = super().data_parallel
        self.q1_model = DDP(
            self.q1_model,
            device_ids=None if device_id is None else [device_id],  # 1 GPU.
            output_device=device_id,
        )
        self.q2_model = DDP(
            self.q2_model,
            device_ids=None if device_id is None else [device_id],  # 1 GPU.
            output_device=device_id,
        )
        return device_id

    def give_min_itr_learn(self, min_itr_learn):
        self.min_itr_learn = min_itr_learn  # From algo.

    def make_env_to_model_kwargs(self, env_spaces):
        return dict(
            input_size=env_spaces.observation.shape.state[0],
            output_size=env_spaces.action.shape[0],
        )

    def pi(self, model_input, init_rnn_state):
        """Compute action log-probabilities for state/observation, and
        sample new action (with grad).  Uses special ``sample_loglikelihood()``
        method of Gaussian distriution, which handles action squashing
        through this process."""
        model_input = buffer_to(model_input, device=self.device)
        init_rnn_state = buffer_to(init_rnn_state, self.device)
        action, rnn_state = self.model(*model_input, init_rnn_state)
        dist_info = DistInfoStd(mean=action, log_std=action)
        # action, log_pi = self.distribution.sample_loglikelihood(dist_info)
        # log_pi, dist_info = buffer_to((log_pi, dist_info), device="cpu")
        return action.cpu(), dist_info, rnn_state  # Action stays on device for q models.

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        model_inputs = buffer_to((observation, prev_action, prev_reward), device=self.device)
        action, rnn_state = self.model(*model_inputs, self.prev_rnn_state)
        # dist_info = DistInfoStd(mean=mean, log_std=log_std)
        # action = self.distribution.sample(dist_info)

        prev_rnn_state = self.prev_rnn_state or buffer_func(rnn_state, torch.zeros_like)
        # Transpose the rnn_state from [N,B,H] --> [B,N,H] for storage.
        # (Special case, model should always leave B dimension in.)
        prev_rnn_state = buffer_method(prev_rnn_state, "transpose", 0, 1)
        prev_rnn_state = buffer_to(prev_rnn_state, device="cpu")
        agent_info = AgentInfo(action=action, prev_rnn_state=prev_rnn_state)

        action, agent_info = buffer_to((action, agent_info), device="cpu")
        self.advance_rnn_state(rnn_state)
        return AgentStep(action=action, agent_info=agent_info)

    def update_target(self, tau=1):
        update_state_dict(self.target_q1_model, self.q1_model.state_dict(), tau)
        update_state_dict(self.target_q2_model, self.q2_model.state_dict(), tau)

    @property
    def models(self):
        return Models(pi=self.model, q1=self.q1_model, q2=self.q2_model)

    def pi_parameters(self):
        return self.model.parameters()

    def train_mode(self, itr):
        super().train_mode(itr)

    def sample_mode(self, itr):
        super().sample_mode(itr)

    def eval_mode(self, itr):
        super().eval_mode(itr)
