from learning_to_be_taught.vmpo.async_vmpo import AsyncVMPO, OptInfo
from rlpyt.utils.buffer import buffer_to, buffer_method
from learning_to_be_taught.vmpo.v_mpo import VMPO
import torch
from rlpyt.algos.utils import (discount_return, generalized_advantage_estimation,
                               valid_from_done)
from learning_to_be_taught.vmpo.popart_normalization import PopArtLayer
from learning_to_be_taught.recurrent_sac.pc_grad import PCGradSGD


class MultitaskVmpoMixin:
    def __init__(self, num_tasks, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pop_art_normalizer = PopArtLayer(output_features=num_tasks)
        self.num_tasks = num_tasks

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
                                                  opt_info=opt_info,
                                                  task=batch.agent_inputs.observation.task_id)
                                                  # demonstration_phase=batch.agent_inputs.observation.demonstration_phase)
            # if len(agent_output) > 3:
            #     aux_loss = agent_output[3]
            #     loss = loss + aux_loss.cpu()
            loss.backward()
            self.optimizer.step()
            self.clamp_lagrange_multipliers()

            opt_info.loss.append(loss.item())
            opt_info.optim_buffer_wait_time.append(buffer_wait_time)
            self.update_counter += 1
        return opt_info

    def process_returns(self, reward, done, value_prediction,
                        action, dist_info, old_dist_info, opt_info, task=None, demonstration_phase=None):
        done = done.type(reward.dtype)
        if self.pop_art_reward_normalization:
            unnormalized_value = value_prediction
            value_prediction, normalized_value = self.pop_art_normalizer(value_prediction, task=task)

        bootstrap_value = value_prediction[-1]
        reward, value_prediction, done, task = reward[:-1], value_prediction[:-1], done[:-1], task[:-1]

        return_ = discount_return(reward, done, bootstrap_value.detach(), self.discount)
        # advantage, return_ = generalized_advantage_estimation(
        #     reward, value_prediction, done, bootstrap_value, self.discount, self.gae_lambda)

        if self.pop_art_reward_normalization:
            self.pop_art_normalizer.update_parameters(return_.unsqueeze(-1).expand(*return_.shape, self.num_tasks),
                                                              torch.nn.functional.one_hot(task, self.num_tasks))
            _, normalized_value = self.pop_art_normalizer(unnormalized_value[:-1], task=task)
            return_ = self.pop_art_normalizer.normalize(return_, task=task).detach()
            advantage = return_ - normalized_value.detach()
            value_prediction = normalized_value
            opt_info.normalized_return.append(return_.numpy())
        else:
            advantage = return_ - value_prediction.detach()

        valid = valid_from_done(done)  # Recurrent: no reset during training.
        # advantage = advantage * (1 - demonstration_phase[:-1].int()) # mask all steps during demonstration
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


class MultitaskAsyncVMPO(MultitaskVmpoMixin, AsyncVMPO):
    pass

class MultitaskVMPO(MultitaskVmpoMixin, VMPO):
    pass