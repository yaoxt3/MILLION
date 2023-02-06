
import numpy as np
import torch

from rlpyt.agents.base import (AgentStep, BaseAgent, RecurrentAgentMixin,
                               AlternatingRecurrentAgentMixin)
from rlpyt.agents.pg.base import AgentInfo, AgentInfoRnn
from rlpyt.distributions.gaussian import Gaussian, DistInfoStd
from rlpyt.utils.buffer import buffer_to, buffer_func, buffer_method
from rlpyt.agents.pg.gaussian import GaussianPgAgent, RecurrentGaussianPgAgentBase
from rlpyt.agents.pg.mujoco import MujocoMixin

# MIN_STD = 1e-6

class VmpoAgentBase(MujocoMixin, RecurrentGaussianPgAgentBase):
    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        """
        Compute policy's action distribution from inputs, and sample an
        action. Calls the model to produce mean, log_std, value estimate, and
        next recurrent state.  Moves inputs to device and returns outputs back
        to CPU, for the sampler.  Advances the recurrent state of the agent.
        (no grad)
        """
        agent_inputs = buffer_to((observation, prev_action, prev_reward),
                                 device=self.device)
        mu, log_std, value, rnn_state = self.model(*agent_inputs, self.prev_rnn_state)
        dist_info = DistInfoStd(mean=mu, log_std=log_std)
        action = self.distribution.sample(dist_info)
        # print(f'action {action[0]} mean: {dist_info.mean[0]}')
        # Model handles None, but Buffer does not, make zeros if needed:
        # prev_rnn_state = self.prev_rnn_state or buffer_func(rnn_state, torch.zeros_like)
        if self.prev_rnn_state is None:
            prev_rnn_state = buffer_func(rnn_state, torch.zeros_like)
        else:
            prev_rnn_state = self.prev_rnn_state

        # Transpose the rnn_state from [N,B,H] --> [B,N,H] for storage.
        # (Special case: model should always leave B dimension in.)
        prev_rnn_state = buffer_method(prev_rnn_state, "transpose", 0, 1)
        agent_info = AgentInfoRnn(dist_info=dist_info, value=value,
                                  prev_rnn_state=prev_rnn_state)
        action, agent_info = buffer_to((action, agent_info), device="cpu")
        self.advance_rnn_state(rnn_state)  # Keep on device.
        return AgentStep(action=action, agent_info=agent_info)

    def eval_mode(self, itr):
        super().eval_mode(itr)
        # print('eval mode #################')
        self.distribution.set_std(0)

    def sample_mode(self, itr):
        super().sample_mode(itr)
        # print("sample mode ############################")
        self.distribution.set_std(None)

class VmpoAgent(RecurrentAgentMixin, VmpoAgentBase):
    pass


class AlternatingVmpoAgent(AlternatingRecurrentAgentMixin,
                                          VmpoAgentBase):
    pass
