import torch
import numpy as np
import learning_to_be_taught.vmpo.utils as utils
from rlpyt.algos.base import RlAlgorithm
from learning_to_be_taught.vmpo.popart_normalization import PopArtLayer
from rlpyt.algos.pg.base import OptInfo
from rlpyt.agents.base import AgentInputs
from rlpyt.utils.tensor import valid_mean
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.collections import namedarraytuple, namedtuple
from rlpyt.utils.misc import iterate_mb_idxs
from rlpyt.utils.buffer import buffer_to, buffer_method
from rlpyt.algos.utils import (discount_return, generalized_advantage_estimation, valid_from_done)

LossInputs = namedarraytuple("LossInputs",
                             ["agent_inputs", "action", "return_", "advantage", "valid", "old_dist_info"])

OptInfo = namedtuple("OptInfo", ["loss", 'pi_loss', 'eta_loss', 'alpha_loss', 'value_loss',
                                 'alpha_mu_loss', 'alpha_sigma_loss',
                                 'mu_kl', 'sigma_kl', 'advantage', 'normalized_return',
                                 'alpha', 'eta', 'alpha_mu', 'alpha_sigma',
                                 'pi_mu', 'pi_log_std', 'policy_kl',
                                 "entropy", "perplexity"])

SamplesToBuffer = namedarraytuple("SamplesToBuffer",
                                  ["agent_inputs", "action", "return_", "advantage", "valid", "old_dist_info"])
SamplesToBufferTl = namedarraytuple("SamplesToBufferTl",
                                    SamplesToBuffer._fields + ("timeout",))

SamplesToBufferRnn = namedarraytuple("SamplesToBufferRnn", SamplesToBuffer._fields + ("prev_rnn_state",))


class VMPO(RlAlgorithm):
    opt_info_fields = tuple(f for f in OptInfo._fields)  # copy
    bootstrap_value = False  # Tells the sampler it needs Value(State')

    def __init__(
            self,
            discount=0.99,
            learning_rate=1e-4,
            T_target_steps=100,
            bootstrap_with_online_model=False,
            OptimCls=torch.optim.Adam,
            pop_art_reward_normalization=True,
            optim_kwargs=None,
            initial_optim_state_dict=None,
            minibatches=1,
            epochs=1,
            gae_lambda=0.97,
            discrete_actions=False,
            epsilon_eta=0.01,
            epsilon_alpha=0.01,
            initial_eta=1.0,
            initial_alpha=5.0,
            initial_alpha_mu=1.0,
            initial_alpha_sigma=1.0,
            epsilon_alpha_mu=0.0075,
            epsilon_alpha_sigma=1e-5,
    ):
        """Saves input settings."""
        if optim_kwargs is None:
            optim_kwargs = dict()
        self.pop_art_normalizer = PopArtLayer()
        save__init__args(locals())

    def initialize(self, agent, n_itr, batch_spec, mid_batch_reset=False,
                   examples=None, world_size=1, rank=0):
        self.agent = agent
        self.alpha = torch.autograd.Variable(torch.ones(1) * self.initial_alpha, requires_grad=True)
        self.alpha_mu = torch.autograd.Variable(torch.ones(1) * self.initial_alpha_mu, requires_grad=True)
        self.alpha_sigma = torch.autograd.Variable(torch.ones(1) * self.initial_alpha_sigma, requires_grad=True)
        self.eta = torch.autograd.Variable(torch.ones(1) * self.initial_eta, requires_grad=True)

        self.optimizer = self.OptimCls(list(self.agent.parameters()) +
                                       list(self.pop_art_normalizer.parameters()) +
                                       [self.alpha, self.alpha_mu, self.alpha_sigma, self.eta],
                                       lr=self.learning_rate, **self.optim_kwargs)
        if self.initial_optim_state_dict is not None:
            self.load_optim_state_dict(self.initial_optim_state_dict)
        self.n_itr = n_itr
        self.batch_spec = batch_spec
        self.mid_batch_reset = mid_batch_reset
        self.rank = rank
        self.world_size = world_size
        self._batch_size = self.batch_spec.size // self.minibatches  # For logging.

    def process_returns(self, reward, done, value_prediction,
                        action, dist_info, old_dist_info, opt_info):
        done = done.type(reward.dtype)
        if self.pop_art_reward_normalization:
            unnormalized_value = value_prediction
            value_prediction, normalized_value = self.pop_art_normalizer(value_prediction)
        else:
            value_prediction = value_prediction.squeeze(-1)

        bootstrap_value = value_prediction[-1]
        reward, value_prediction, done = reward[:-1], value_prediction[:-1], done[:-1]

        return_ = discount_return(reward, done, bootstrap_value.detach(), self.discount)
        if self.pop_art_reward_normalization:
            self.pop_art_normalizer.update_parameters(return_.unsqueeze(-1),
                                                      torch.ones_like(return_.unsqueeze(-1)))
            _, normalized_value = self.pop_art_normalizer(unnormalized_value[:-1])
            return_ = self.pop_art_normalizer.normalize(return_)
            advantage = return_ - normalized_value.detach()
            value_prediction = normalized_value
            opt_info.normalized_return.append(return_.numpy())
        else:
            advantage = return_ - value_prediction.detach()

        valid = valid_from_done(done)  # Recurrent: no reset during training.
        opt_info.advantage.append(advantage.numpy())

        loss, opt_info = self.loss(dist_info=dist_info[:-1],
                                   value=value_prediction,
                                   action=action[:-1],
                                   return_=return_,
                                   advantage=advantage.detach(),
                                   valid=valid,
                                   old_dist_info=old_dist_info[:-1],
                                   opt_info=opt_info)
        return loss, opt_info

    def optimize_agent(self, itr, samples=None, sampler_itr=None):
        """
        Train the agent, for multiple epochs over minibatches taken from the
        input samples.  Organizes agent inputs from the training data, and
        moves them to device (e.g. GPU) up front, so that minibatches are
        formed within device, without further data transfer.
        """
        opt_info = OptInfo(*([] for _ in range(len(OptInfo._fields))))
        agent_inputs = AgentInputs(  # Move inputs to device once, index there.
            observation=samples.env.observation,
            prev_action=samples.agent.prev_action,
            prev_reward=samples.env.prev_reward,
        )
        agent_inputs = buffer_to(agent_inputs, device=self.agent.device)
        init_rnn_states = buffer_to(samples.agent.agent_info.prev_rnn_state[0], device=self.agent.device)
        T, B = samples.env.reward.shape[:2]
        mb_size = B // self.minibatches
        for _ in range(self.epochs):
            for idxs in iterate_mb_idxs(B, mb_size, shuffle=True):
                self.optimizer.zero_grad()
                init_rnn_state = buffer_method(init_rnn_states[idxs], "transpose", 0, 1)
                dist_info, value, _, _  = self.agent(*agent_inputs[:, idxs], init_rnn_state)
                loss, opt_info = self.process_returns(samples.env.reward[:, idxs],
                                                      done=samples.env.done[:, idxs],
                                                      value_prediction=value.cpu(),
                                                      action=samples.agent.action[:, idxs],
                                                      dist_info=dist_info,
                                                      old_dist_info=samples.agent.agent_info.dist_info[:, idxs],
                                                      opt_info=opt_info)
                loss.backward()
                self.optimizer.step()
                self.clamp_lagrange_multipliers()
                opt_info.loss.append(loss.item())
                self.update_counter += 1
        return opt_info

    def loss(self, dist_info, value, action, return_, advantage, valid, old_dist_info, opt_info):
        T, B = tuple(action.shape[:2])
        advantage = advantage.clamp(min=-40, max=40)  # clamp due to numerical instability of exp function

        num_valid = valid.sum().type(torch.int32)
        top_advantages, top_advantages_indeces = torch.topk((advantage.masked_fill(~valid.bool(), float('-inf'))).reshape(T * B),
                                                            num_valid // 2)
        advantage_mask = torch.zeros_like(advantage.view(T * B))
        advantage_mask[top_advantages_indeces] = 1
        advantage_mask = advantage_mask.reshape(T, B)

        log_advantage_sum = torch.logsumexp(top_advantages / self.eta, dim=0)
        psi = torch.exp((advantage / self.eta) - log_advantage_sum)
        value_error = 0.5 * (value - return_) ** 2
        value_loss = valid_mean(value_error, valid)
        eta_loss = self.eta * self.epsilon_eta + self.eta * (log_advantage_sum - torch.log(0.5 * num_valid))

        if self.discrete_actions:
            pi_loss, alpha_loss, opt_info = self.discrete_actions_loss(advantage_mask, psi, action, dist_info,
                                                                       old_dist_info, valid, opt_info)
            loss = pi_loss + value_loss + eta_loss + alpha_loss
        else:
            pi_loss, alpha_loss, opt_info = self.continuous_actions_loss(advantage_mask, psi, action, dist_info,
                                                                         old_dist_info, valid, opt_info)
            loss = pi_loss + value_loss + eta_loss + alpha_loss

        opt_info.pi_loss.append(pi_loss.item())
        opt_info.eta_loss.append(eta_loss.item())
        opt_info.value_loss.append(value_loss.item())
        opt_info.eta.append(self.eta.item())
        return loss, opt_info

    def discrete_actions_loss(self, advantage_mask, phi, action, dist_info, old_dist_info, valid, opt_info):
        dist = self.agent.distribution
        pi_loss = - torch.sum(advantage_mask * (phi.detach() * dist.log_likelihood(action.contiguous(), dist_info)))
        policy_kl = dist.kl(old_dist_info, dist_info)
        alpha_loss = valid_mean(
            self.alpha * (self.epsilon_alpha - policy_kl.detach()) + self.alpha.detach() * policy_kl, valid)
        opt_info.alpha_loss.append(alpha_loss.item())
        opt_info.alpha.append(self.alpha.item())
        opt_info.policy_kl.append(policy_kl.mean().item())
        opt_info.entropy.append(dist.entropy(dist_info).mean().item())
        return pi_loss, alpha_loss, opt_info

    def continuous_actions_loss(self, advantage_mask, phi, action, dist_info, old_dist_info, valid, opt_info):
        d = np.prod(action.shape[-1])
        distribution = torch.distributions.normal.Normal(loc=dist_info.mean, scale=dist_info.log_std)
        pi_loss = - torch.sum(advantage_mask * (phi.detach() * distribution.log_prob(action).sum(dim=-1)))
        # pi_loss = - torch.sum(advantage_mask * (phi.detach() * self.agent.distribution.log_likelihood(action, dist_info)))
        new_std = dist_info.log_std
        old_std = old_dist_info.log_std
        old_covariance = torch.diag_embed(old_std)
        old_covariance_inverse = torch.diag_embed(1 / old_std)
        new_covariance_inverse = torch.diag_embed(1 / new_std)
        old_covariance_determinant = torch.prod(old_std, dim=-1)
        new_covariance_determinant = torch.prod(new_std, dim=-1)

        mu_kl = 0.5 * utils.batched_quadratic_form(dist_info.mean - old_dist_info.mean, old_covariance_inverse)
        trace = utils.batched_trace(torch.matmul(new_covariance_inverse, old_covariance))
        sigma_kl = 0.5 * (trace - d + torch.log(new_covariance_determinant / old_covariance_determinant))
        alpha_mu_loss = valid_mean(
            self.alpha_mu * (self.epsilon_alpha_mu - mu_kl.detach()) + self.alpha_mu.detach() * mu_kl, valid)
        alpha_sigma_loss = valid_mean(self.alpha_sigma * (
                self.epsilon_alpha_sigma - sigma_kl.detach()) + self.alpha_sigma.detach() * sigma_kl, valid)
        opt_info.alpha_mu.append(self.alpha_mu.item())
        opt_info.alpha_sigma.append(self.alpha_sigma.item())
        opt_info.alpha_mu_loss.append(alpha_mu_loss.item())
        opt_info.mu_kl.append(valid_mean(mu_kl, valid).item())
        opt_info.sigma_kl.append(valid_mean(sigma_kl, valid).item())
        opt_info.alpha_sigma_loss.append(valid_mean(self.epsilon_alpha_sigma - sigma_kl, valid).item())
        opt_info.pi_mu.append(dist_info.mean.mean().item())
        opt_info.pi_log_std.append(dist_info.log_std.mean().item())
        return pi_loss, alpha_mu_loss + alpha_sigma_loss, opt_info

    def beta_dist_loss(self, advantage_mask, phi, action, dist_info, old_dist_info, valid, opt_info):
        action = (action + 1) / 2
        distribution = torch.distributions.beta.Beta(dist_info.mean, dist_info.log_std)
        old_dist = torch.distributions.beta.Beta(old_dist_info.mean, old_dist_info.log_std)
        pi_loss = - torch.sum(advantage_mask * (phi.detach() * distribution.log_prob(action).sum(dim=-1)))
        kl = torch.distributions.kl_divergence(old_dist, distribution).sum(dim=-1)
        alpha_loss = valid_mean(self.alpha * (self.epsilon_alpha - kl.detach()) + self.alpha.detach() * kl, valid)
        entropy = valid_mean(distribution.entropy().sum(dim=-1), valid)
        alpha_loss -= 0.01 * entropy

        mode = self.agent.beta_dist_mode(old_dist_info.mean, old_dist_info.log_std)
        opt_info.alpha.append(self.alpha.item())
        opt_info.policy_kl.append(kl.mean().item())
        opt_info.pi_mu.append(mode.mean().item())
        opt_info.pi_log_std.append(old_dist.entropy().sum(dim=-1).mean().item())
        return pi_loss, alpha_loss, opt_info

    def clamp_lagrange_multipliers(self):
        """
        As described in the paper alpha and eta are lagrange multipliers that must be positive. That's why
        they are clamped after every update
        """
        with torch.no_grad():
            self.alpha.clamp_min_(1e-8)
            self.alpha_mu.clamp_min_(1e-8)
            self.alpha_sigma.clamp_min_(1e-8)
            self.eta.clamp_min_(1e-8)

    def optim_state_dict(self):
        return dict(
            optimizer=self.optimizer.state_dict(),
            eta=self.eta,
            alpha=self.alpha,
            alpha_mu=self.alpha_mu,
            alpha_sigma=self.alpha_sigma,
            pop_art_layer=self.pop_art_normalizer.state_dict()
        )

    def load_optim_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.pop_art_normalizer.load_state_dict(state_dict['pop_art_layer'])
        self.eta.data = state_dict['eta']
        self.alpha.data = state_dict['alpha']
        self.alpha_mu.data = state_dict['alpha_mu']
        self.alpha_sigma.data = state_dict['alpha_sigma']
