import numpy as np
import torch
from torch.utils.data import DataLoader
from learning_to_be_taught.sequence_replay import TorchDataset
from collections import namedtuple

import time
from rlpyt.agents.base import AgentInputs
from rlpyt.algos.base import RlAlgorithm
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.logging import logger
from rlpyt.replays.non_sequence.uniform import (UniformReplayBuffer,
                                                AsyncUniformReplayBuffer)
from rlpyt.replays.sequence.uniform import UniformSequenceReplayBuffer, AsyncUniformSequenceReplayBuffer
from rlpyt.replays.non_sequence.time_limit import (TlUniformReplayBuffer,
                                                   AsyncTlUniformReplayBuffer)
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.buffer import buffer_to, buffer_method, torchify_buffer
from rlpyt.distributions.gaussian import Gaussian
from rlpyt.distributions.gaussian import DistInfo as GaussianDistInfo
from rlpyt.utils.tensor import valid_mean
from rlpyt.algos.utils import valid_from_done
from rlpyt.algos.qpg.sac import SAC
from learning_to_be_taught.recurrent_sac.recurrent_sac_agent import AgentInputs, QRnnState, QValues

OptInfo = namedtuple("OptInfo",
                     ["q1Loss", "q2Loss", "piLoss",
                      "q1GradNorm", "q2GradNorm", "piGradNorm", 'baseModelGradNorm',
                      "q1", "q2", "piMu", "piLogStd", "qMeanDiff", "alpha", 'learning_rate'])
SamplesToBuffer = namedarraytuple("SamplesToBuffer",
                                  ["observation", "action", "reward", "done"])
SamplesToBufferTl = namedarraytuple("SamplesToBufferTl",
                                    SamplesToBuffer._fields + ("timeout",))

SamplesToBufferRnn = namedarraytuple("SamplesToBufferRnn", SamplesToBuffer._fields + ("prev_rnn_state",))


class EfficientRecurrentSac(SAC):
    """Soft actor critic algorithm, training from a replay buffer."""

    opt_info_fields = tuple(f for f in OptInfo._fields)  # copy

    def __init__(
            self,
            warmup_T=0,
            batch_T=1,
            store_rnn_state_interval=1,  # 0 for none, 1 for all.
            max_learning_rate=1e-5,
            warmup_steps=5e3,
            mixed_precision=True,
            alternative_pi_loss=True,
            **kwargs
    ):
        save__init__args(locals())
        self.update_counter = 0
        super().__init__(**kwargs)

    def initialize(self, **kwargs):
        super().initialize(**kwargs)
        self.updates_per_optimize = max(1, round(self.updates_per_optimize / self.batch_T))
        logger.log('divided updates per iteration by ' + str(self.batch_T) +
                   '. New number of updates per iteration is ' + str(self.updates_per_optimize))

    def initialize_replay_buffer(self, examples, batch_spec, async_=False):
        """
        Allocates replay buffer using examples and with the fields in `SamplesToBuffer`
        namedarraytuple.
        """
        """Similar to DQN but uses replay buffers which return sequences, and
        stores the agent's recurrent state."""
        example_to_buffer = SamplesToBuffer(
            observation=examples["observation"],
            action=examples["action"],
            reward=examples["reward"],
            done=examples["done"],
        )
        if self.store_rnn_state_interval > 0:
            example_to_buffer = SamplesToBufferRnn(*example_to_buffer,
                                                   prev_rnn_state=examples["agent_info"].prev_rnn_state)

        replay_kwargs = dict(
            example=example_to_buffer,
            size=self.replay_size,
            B=batch_spec.B,
            discount=self.discount,
            n_step_return=self.n_step_return,
            rnn_state_interval=self.store_rnn_state_interval,
            # batch_T fixed for prioritized, (relax if rnn_state_interval=1 or 0).
            batch_T=self.batch_T + self.warmup_T,
        )
        ReplayCls = (AsyncUniformSequenceReplayBuffer if async_ else UniformSequenceReplayBuffer)
        self.replay_buffer = ReplayCls(**replay_kwargs)
        return self.replay_buffer

    def samples_to_buffer(self, samples):
        # samples_to_buffer = super().samples_to_buffer(samples)
        samples_to_buffer = SamplesToBuffer(
            observation=samples.env.observation,
            action=samples.agent.action,
            reward=samples.env.reward,
            done=samples.env.done,
        )
        if self.store_rnn_state_interval > 0:
            samples_to_buffer = SamplesToBufferRnn(*samples_to_buffer,
                                                   prev_rnn_state=samples.agent.agent_info.prev_rnn_state)
        return samples_to_buffer

    def optim_initialize(self, rank=0):
        """Called in initilize or by async runner after forking sampler."""
        self.rank = rank
        self.pi_optimizer = self.OptimCls(self.agent.pi_parameters(),
                                          lr=self.learning_rate, **self.optim_kwargs)
        self.base_model_optimizer = self.OptimCls(self.agent.base_model_parameters(), lr=self.learning_rate,
                                                  **self.optim_kwargs)

        self.pi_head_optimizer = self.OptimCls(self.agent.pi_head_parameters(),
                                               lr=self.learning_rate, **self.optim_kwargs)
        self.q1_optimizer = self.OptimCls(self.agent.q1_parameters(),
                                          lr=self.learning_rate, **self.optim_kwargs)
        self.q2_optimizer = self.OptimCls(self.agent.q2_parameters(),
                                          lr=self.learning_rate, **self.optim_kwargs)
        self.q_optimizer = self.OptimCls(list(self.agent.q1_parameters()) + list(self.agent.q2_parameters()),
                                         lr=self.learning_rate)
        self._log_alpha = torch.zeros(1, requires_grad=True, device=self.agent.device)
        self._alpha = torch.exp(self._log_alpha.detach())
        self.alpha_optimizer = self.OptimCls((self._log_alpha,),
                                             lr=self.learning_rate, **self.optim_kwargs)

        self.grad_scaler = torch.cuda.amp.GradScaler()
        if self.target_entropy == "auto":
            self.target_entropy = -np.prod(self.agent.env_spaces.action.shape)
        if self.initial_optim_state_dict is not None:
            self.load_optim_state_dict(self.initial_optim_state_dict)
        if self.action_prior == "gaussian":
            self.action_prior_distribution = Gaussian(
                dim=np.prod(self.agent.env_spaces.action.shape), std=1.)

    def optimize_agent(self, itr, samples=None, sampler_itr=None):
        """
        Extracts the needed fields from input samples and stores them in the 
        replay buffer.  Then samples from the replay buffer to train the agent
        by gradient updates (with the number of updates determined by replay
        ratio, sampler batch size, and training batch size).
        """
        itr = itr if sampler_itr is None else sampler_itr  # Async uses sampler_itr.
        if samples is not None:
            samples_to_buffer = self.samples_to_buffer(samples)
            self.replay_buffer.append_samples(samples_to_buffer)
        opt_info = OptInfo(*([] for _ in range(len(OptInfo._fields))))
        if itr < self.min_itr_learn:
            return opt_info
        new_learning_rate = self.update_learning_rate(self.update_counter)
        opt_info.learning_rate.append(new_learning_rate)
        if self.update_counter % 100 == 0 or self.data_loader is None:
            self.data_loader = iter(
                DataLoader(TorchDataset(self.replay_buffer, batch_size=self.batch_size), num_workers=4, batch_size=None,
                           collate_fn=lambda x: x))
        for _ in range(self.updates_per_optimize):
            # samples_from_replay = self.replay_buffer.sample_batch(self.batch_size)
            samples_from_replay = next(self.data_loader)
            if self.mixed_precision and self.agent.device.type == 'cuda':
                losses, values, grad_norms = self.mixed_precision_update(samples_from_replay)
            else:
                losses, values, grad_norms = self.full_precision_update(samples_from_replay)

            alpha_loss = losses[-1]
            if alpha_loss is not None:
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                self._alpha = torch.exp(self._log_alpha.detach())

            losses = tuple(loss.cpu() for loss in losses)
            values = tuple(value.cpu() for value in values)
            self.append_opt_info_(opt_info, losses, grad_norms, values)
            self.update_counter += 1
            if self.update_counter % self.target_update_interval == 0:
                self.agent.update_target(self.target_update_tau)

        return opt_info

    def full_precision_update(self, samples):
        losses, values = self.loss(samples)
        q1_loss, q2_loss, pi_loss, alpha_loss = losses
        self.pi_head_optimizer.zero_grad()
        self.base_model_optimizer.zero_grad()
        pi_loss.backward(retain_graph=True)
        pi_grad_norm = torch.nn.utils.clip_grad_norm_(self.agent.pi_parameters(), self.clip_grad_norm)

        self.q1_optimizer.zero_grad()
        q1_loss.backward(retain_graph=True)
        q1_grad_norm = torch.nn.utils.clip_grad_norm_(self.agent.q1_parameters(), self.clip_grad_norm)

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        q2_grad_norm = torch.nn.utils.clip_grad_norm_(self.agent.q2_parameters(), self.clip_grad_norm)
        base_model_grad_norm = torch.nn.utils.clip_grad_norm_(self.agent.base_model_parameters(), self.clip_grad_norm)

        self.base_model_optimizer.step()
        self.pi_head_optimizer.step()
        self.q1_optimizer.step()
        self.q2_optimizer.step()
        grad_norms = (q1_grad_norm, q2_grad_norm, pi_grad_norm, base_model_grad_norm)
        return losses, values, grad_norms

    def mixed_precision_update(self, samples):
        with torch.cuda.amp.autocast():
            losses, values = self.loss(samples)
            q1_loss, q2_loss, pi_loss, alpha_loss = losses
            q_loss = q1_loss + q2_loss

        self.pi_head_optimizer.zero_grad()
        self.base_model_optimizer.zero_grad()
        self.grad_scaler.scale(pi_loss).backward(retain_graph=True)

        # self.q1_optimizer.zero_grad()
        self.q_optimizer.zero_grad()
        self.grad_scaler.scale(q_loss).backward()
        # self.grad_scaler.scale(q1_loss).backward(retain_graph=True)

        # self.q2_optimizer.zero_grad()
        # self.grad_scaler.scale(q2_loss).backward()

        self.grad_scaler.unscale_(self.pi_optimizer)
        # self.grad_scaler.unscale_(self.q1_optimizer)
        # self.grad_scaler.unscale_(self.q2_optimizer)
        self.grad_scaler.unscale_(self.q_optimizer)
        self.grad_scaler.unscale_(self.base_model_optimizer)

        pi_grad_norm = torch.nn.utils.clip_grad_norm_(self.agent.pi_parameters(), self.clip_grad_norm)
        q1_grad_norm = torch.nn.utils.clip_grad_norm_(self.agent.q1_parameters(), self.clip_grad_norm)
        q2_grad_norm = torch.nn.utils.clip_grad_norm_(self.agent.q2_parameters(), self.clip_grad_norm)
        base_model_grad_norm = torch.nn.utils.clip_grad_norm_(self.agent.base_model_parameters(), self.clip_grad_norm)

        self.grad_scaler.step(self.base_model_optimizer)
        self.grad_scaler.step(self.pi_head_optimizer)
        # self.grad_scaler.step(self.q1_optimizer)
        # self.grad_scaler.step(self.q2_optimizer)
        self.grad_scaler.step(self.q_optimizer)
        self.grad_scaler.update()

        losses = tuple((val.cpu() for val in losses))
        grad_norms = (q1_grad_norm, q2_grad_norm, pi_grad_norm, base_model_grad_norm)
        return losses, values, grad_norms

    def warmup(self, samples):
        if self.store_rnn_state_interval == 0:
            init_rnn_state = None
        else:
            # [B,N,H]-->[N,B,H] cudnn.
            init_rnn_state = buffer_method(samples.init_rnn_state, "transpose", 0, 1)
            init_rnn_state = buffer_method(init_rnn_state, "contiguous")
        q_rnn_states = QRnnState(q1=init_rnn_state.q1, q2=init_rnn_state.q2,
                                 target_q1=init_rnn_state.q1, target_q2=init_rnn_state.q2)
        pi_rnn_state = init_rnn_state.pi
        if self.warmup_T > 0:
            warmup_slice = slice(None, self.warmup_T)  # Same for agent and target.
            warmup_inputs = AgentInputs(
                observation=samples.all_observation[warmup_slice],
                prev_action=samples.all_action[warmup_slice],
                prev_reward=samples.all_reward[warmup_slice],
            )
            # warmup_inputs = buffer_to(pi_warmup_inputs, self.agent.device)

            with torch.no_grad():
                _, _, _, pi_rnn_state = self.agent.pi(warmup_inputs, pi_rnn_state)
                _, q_rnn_states = self.agent.all_q(warmup_inputs, q_rnn_states)
            # Recommend aligning sampling batch_T and store_rnn_interval with
            # warmup_T (and no mid_batch_reset), so that end of trajectory
            # during warmup leads to new trajectory beginning at start of
            # training segment of replay.
            # warmup_invalid_mask = valid_from_done(samples.done[:self.warmup_T])[-1] == 0  # [B]
            # rnn_state[:, warmup_invalid_mask] = 0  # [N,B,H] (cudnn)
            # target_rnn_state[:, warmup_invalid_mask] = 0

        return pi_rnn_state, q_rnn_states

    def loss(self, samples):
        """
        Computes losses for twin Q-values against the min of twin target Q-values
        and an entropy term.  Computes reparameterized policy loss, and loss for
        tuning entropy weighting, alpha.  
        
        Input samples have leading batch dimension [B,..] (but not time).
        """
        samples = buffer_to(samples, self.agent.device)
        pi_rnn_state, q_rnn_states = self.warmup(samples)
        wT, bT, nsr = self.warmup_T, self.batch_T, self.n_step_return
        agent_slice = slice(self.warmup_T, self.warmup_T + self.batch_T)
        agent_inputs = AgentInputs(
            observation=samples.all_observation[agent_slice],
            prev_action=samples.all_action[agent_slice],
            prev_reward=samples.all_reward[agent_slice],
        )
        target_slice = slice(self.warmup_T, None)
        target_inputs = AgentInputs(
            observation=samples.all_observation[target_slice],
            prev_action=samples.all_action[target_slice],
            prev_reward=samples.all_reward[target_slice],
        )
        actions = samples.all_action[wT + 1:wT + 1 + bT]  # CPU.
        return_ = samples.return_[wT:wT + bT]
        done_n = samples.done_n[wT:wT + bT]
        # import pdb; pdb.set_trace()
        agent_features = self.agent.features(agent_inputs, pi_rnn_state)
        # agent_features = target_features[:bT]
        q1, q2 = self.agent.q(agent_features, agent_inputs, q_rnn_states.q1, q_rnn_states.q2, actions)
        with torch.no_grad():
            target_features = self.agent.target_features(target_inputs, pi_rnn_state)
            target_actions, log_pi, _, _ = self.agent.pi(target_features, target_inputs, pi_rnn_state)
            target_q1, target_q2 = self.agent.target_q(target_features, target_inputs, q_rnn_states.target_q1,
                                                       q_rnn_states.target_q2,
                                                       target_actions)

        min_target_q = torch.min(target_q1, target_q2)
        target_value = min_target_q - self._alpha * log_pi
        disc = self.discount ** self.n_step_return
        y = (self.reward_scale * return_ +
             (1 - done_n.float()) * disc * target_value[self.n_step_return:])

        valid = valid_from_done(samples.done[wT:])  # 0 after first done.
        q1_loss = 0.5 * valid_mean((y - q1) ** 2, valid)
        q2_loss = 0.5 * valid_mean((y - q2) ** 2, valid)

        new_actions, log_pi, (pi_mean, pi_log_std), _ = self.agent.pi(agent_features, agent_inputs, pi_rnn_state)
        log_target1, log_target2 = self.agent.q(agent_features.detach(), agent_inputs, q_rnn_states.q1, q_rnn_states.q2,
                                                new_actions)
        prior_log_pi = self.get_action_prior(new_actions[:-1])

        if self.alternative_pi_loss:
            pi_losses = self._alpha * log_pi - log_target1 - log_target2 - prior_log_pi
        else:
            min_log_target = torch.min(log_target1, log_target2)
            pi_losses = self._alpha * log_pi - min_log_target - prior_log_pi
        pi_loss = valid_mean(pi_losses, valid)

        if self.target_entropy is not None:
            alpha_losses = - self._log_alpha * (log_pi.detach() + self.target_entropy)
            alpha_loss = valid_mean(alpha_losses, valid)
        else:
            alpha_loss = None

        losses = (q1_loss, q2_loss, pi_loss, alpha_loss)
        values = tuple(val.detach() for val in (q1, q2, pi_mean, pi_log_std))
        return losses, values

    def update_learning_rate(self, itr):
        if itr < self.warmup_steps:
            new_learning_rate = self.max_learning_rate * (itr / self.warmup_steps)
        else:
            new_learning_rate = self.max_learning_rate * (0.9999 ** itr)

        new_learning_rate = self.max_learning_rate
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        for param_group in self.base_model_optimizer.param_groups:
            param_group['lr'] = new_learning_rate / 4

        for param_group in self.pi_optimizer.param_groups:
            param_group['lr'] = new_learning_rate

        for param_group in self.q1_optimizer.param_groups:
            param_group['lr'] = new_learning_rate

        for param_group in self.q2_optimizer.param_groups:
            param_group['lr'] = new_learning_rate

        return new_learning_rate

    def append_opt_info_(self, opt_info, losses, grad_norms, values):
        """In-place."""
        q1_grad_norm, q2_grad_norm, pi_grad_norm, base_model_grad_norm = grad_norms
        super().append_opt_info_(opt_info, losses, (q1_grad_norm, q2_grad_norm, pi_grad_norm), values)
        opt_info.baseModelGradNorm.append(torch.tensor(base_model_grad_norm).item())
