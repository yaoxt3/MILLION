from rlpyt.distributions.gaussian import Gaussian
import torch
from rlpyt.agents.pg.base import AgentInfo, AgentInfoRnn
from learning_to_be_taught.vmpo.multivariate_gaussian import MultivariateGaussian, DistInfo
from rlpyt.utils.buffer import buffer_to, buffer_func, buffer_method
from rlpyt.agents.pg.mujoco import MujocoMixin, AlternatingRecurrentGaussianPgAgent
from rlpyt.agents.base import (AgentStep, BaseAgent, RecurrentAgentMixin,
                               AlternatingRecurrentAgentMixin)
from torch.distributions.multivariate_normal import MultivariateNormal
from rlpyt.utils.collections import namedarraytuple
from rlpyt.distributions.gaussian import DistInfoStd

DistInfo = namedarraytuple("DistInfo", ["mean", 'std'])



class VmpoAgent(RecurrentAgentMixin, BaseAgent):

    def initialize(self, env_spaces, *args, **kwargs):
        """Extends base method to build Gaussian distribution."""
        super().initialize(env_spaces, *args, **kwargs)
        # self.distribution = MultivariateNormal
        # self.distribution = MultivariateGaussian(
        #     dim=env_spaces.action.shape[0],
        # )
        self.distribution = Gaussian(
            dim=env_spaces.action.shape[0],
            # squash=env_spaces.action.high[0],
            # min_std=MIN_STD,
            # clip=env_spaces.action.high[0],  # Probably +1?
        )

    def __call__(self, observation, prev_action, prev_reward, init_rnn_state):
        """Performs forward pass on training data, for algorithm."""
        model_inputs = buffer_to((observation, prev_action, prev_reward, init_rnn_state),
                                 device=self.device)
        model_output = self.model(*model_inputs)
        mu, std, value, rnn_state = model_output[:4]
        dist_info, value = buffer_to((DistInfoStd(mean=mu, log_std=std), value), device="cpu")
        return dist_info, value, rnn_state, *model_output[4:]

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        agent_inputs = buffer_to((observation, prev_action, prev_reward), device=self.device)
        with torch.cuda.amp.autocast():
            model_output = self.model(*agent_inputs, self.prev_rnn_state)
        mu, std, value, rnn_state = model_output[:4]
        # print(f'alpha {mu} beta {std}')
        dist_info = DistInfoStd(mean=mu, log_std=std)
        # action = self.distribution.sample(dist_info) if self._mode == 'sample' else mu
        # dist = torch.distributions.beta.Beta(mu, std)
        # if self._mode == 'sample':
        #     action = dist.sample()
        # else:
        #     action = self.beta_dist_mode(mu, std)
        # action = action * 2 - 1

        dist = torch.distributions.normal.Normal(loc=mu, scale=std)
        action = dist.sample() if self._mode == 'sample' else mu
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

    def beta_dist_mode(self, alpha, beta):
        return (alpha - 1) / (alpha + beta - 2)

class MujocoVmpoAgent(MujocoMixin, VmpoAgent):
    pass


class AlternatingVmpoAgent(AlternatingRecurrentAgentMixin, MujocoVmpoAgent):
    pass
