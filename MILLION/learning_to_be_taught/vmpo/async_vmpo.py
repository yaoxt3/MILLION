import torch
import time
from learning_to_be_taught.vmpo.v_mpo import VMPO, OptInfo
from rlpyt.agents.base import AgentInputs, AgentInputsRnn
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.buffer import buffer_to, buffer_method
from rlpyt.utils.collections import namedarraytuple, namedtuple
from learning_to_be_taught.vmpo.on_policy_replay import AsyncUniformSequenceReplayBuffer

LossInputs = namedarraytuple("LossInputs",
                             ["dist_info", "value", "action", "return_", "advantage", "valid", "old_dist_info"])

SamplesToBuffer = namedarraytuple("SamplesToBuffer",
                                  ['agent_inputs', "action", "reward", "done", "dist_info"])
SamplesToBufferTl = namedarraytuple("SamplesToBufferTl",
                                    SamplesToBuffer._fields + ("timeout",))

SamplesToBufferRnn = namedarraytuple("SamplesToBufferRnn", SamplesToBuffer._fields + ("prev_rnn_state",))

OptInfo = namedarraytuple("OptInfo", OptInfo._fields + ("optim_buffer_wait_time",))


class AsyncVMPO(VMPO):
    opt_info_fields = tuple(f for f in OptInfo._fields)  # copy

    def __init__(
            self,
            batch_B=64,
            batch_T=40,
            **kwargs
    ):
        super().__init__(**kwargs)
        self._batch_size = batch_B * batch_T
        self.updates_per_optimize = self.T_target_steps
        save__init__args(locals())
        self.replay_ratio = self.epochs  # make sure runner doesn't throttle this algo
        # self.min_steps_learn = 100

    def initialize(self, *args, **kwargs):
        raise NotImplemented()

    def async_initialize(self, agent, sampler_n_itr, batch_spec, mid_batch_reset, examples, **kwargs):
        """Used in async runner only; returns replay buffer allocated in shared
        memory, does not instantiate optimizer. """
        self.agent = agent
        self.initialize_replay_buffer(examples, batch_spec, async_=True)
        self.mid_batch_reset = mid_batch_reset
        return self.replay_buffer

    def optim_initialize(self, rank=0):
        device = 'cpu'  # self.agent.device
        self.alpha = torch.autograd.Variable(torch.ones(1, device=device) * self.initial_alpha, requires_grad=True)
        self.alpha_mu = torch.autograd.Variable(torch.ones(1, device=device) * self.initial_alpha_mu,
                                                requires_grad=True)
        self.alpha_sigma = torch.autograd.Variable(torch.ones(1, device=device) * self.initial_alpha_sigma,
                                                   requires_grad=True)
        self.eta = torch.autograd.Variable(torch.ones(1, device=device) * self.initial_eta, requires_grad=True)

        self.optimizer = self.OptimCls(list(self.agent.parameters()) +
                                       list(self.pop_art_normalizer.parameters()) +
                                       [self.alpha, self.alpha_mu, self.alpha_sigma, self.eta],
                                       lr=self.learning_rate, **self.optim_kwargs)
        if self.initial_optim_state_dict is not None:
            self.load_optim_state_dict(self.initial_optim_state_dict)

    def samples_to_buffer(self, samples):
        """Defines how to add data from sampler into the replay buffer. Called
        in optimize_agent() if samples are provided to that method."""
        samples_to_buffer = SamplesToBufferRnn(
            agent_inputs=AgentInputs(
                observation=samples.env.observation,
                prev_action=samples.agent.prev_action,
                prev_reward=samples.env.prev_reward,
            ),
            action=samples.agent.action,
            reward=samples.env.reward,
            done=samples.env.done,
            dist_info=samples.agent.agent_info.dist_info,
            prev_rnn_state=samples.agent.agent_info.prev_rnn_state
        )
        return samples_to_buffer

    def initialize_replay_buffer(self, examples, batch_spec, async_=False):
        """
        Allocates replay buffer using examples and with the fields in `SamplesToBuffer`
        namedarraytuple.
        """
        """Similar to DQN but uses replay buffers which return sequences, and
        stores the agent's recurrent state."""
        assert batch_spec.T % self.batch_T == 0, 'sampler batch T has to be a multiple of optim batch T'
        example_to_buffer = SamplesToBuffer(
            agent_inputs=AgentInputs(
                observation=examples["observation"],
                prev_action=examples['action'],
                prev_reward=examples['reward'],
            ),
            action=examples["action"],
            reward=examples["reward"],
            done=examples["done"],
            dist_info=examples['agent_info'].dist_info
            # agent_info=examples['agent_info']
        )
        example_to_buffer = SamplesToBufferRnn(*example_to_buffer,
                                               prev_rnn_state=examples["agent_info"].prev_rnn_state)
        replay_kwargs = dict(
            example=example_to_buffer,
            # size=self.replay_size,
            sampler_B=batch_spec.B,
            discount=self.discount,
            batch_T=self.batch_T,
            optim_B=self.batch_B,
            T_target=self.T_target_steps
        )
        self.replay_buffer = AsyncUniformSequenceReplayBuffer(**replay_kwargs)
        return self.replay_buffer

    def optimize_agent(self, itr, samples=None, sampler_itr=None):
        if samples is not None:
            self.replay_buffer.append_samples(self.samples_to_buffer(samples))

        opt_info = OptInfo(*([] for _ in range(len(OptInfo._fields))))

        batch_generator = self.replay_buffer.batch_generator(replay_ratio=self.epochs)
        for batch, init_rnn_state, buffer_wait_time in batch_generator:
            self.optimizer.zero_grad()
            init_rnn_state = buffer_method(init_rnn_state, "transpose", 0, 1)
            agent_output = self.agent(*batch.agent_inputs, init_rnn_state)
            dist_info, value = agent_output[:2]

            loss, opt_info = self.process_returns(reward=batch.reward,
                                                  done=batch.done,
                                                  value_prediction=value,
                                                  action=batch.action,
                                                  dist_info=dist_info,
                                                  old_dist_info=batch.dist_info,
                                                  opt_info=opt_info)
            loss.backward()
            self.optimizer.step()
            self.clamp_lagrange_multipliers()

            opt_info.loss.append(loss.item())
            opt_info.optim_buffer_wait_time.append(buffer_wait_time)
            self.update_counter += 1
        return opt_info
