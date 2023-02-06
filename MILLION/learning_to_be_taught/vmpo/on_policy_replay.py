import numpy as np
import time
from rlpyt.replays.async_ import AsyncReplayBufferMixin
from rlpyt.utils.buffer import torchify_buffer, buffer_from_example, buffer_func
from rlpyt.utils.buffer import buffer_from_example, get_leading_dims
from rlpyt.utils.collections import namedarraytuple


class AsyncUniformSequenceReplayBuffer(AsyncReplayBufferMixin):
    """Replays sequences with starting state chosen uniformly randomly.
    """

    def __init__(self, example, sampler_B, optim_B, batch_T, discount=1, n_step_return=1, T_target=100):
        super().__init__()
        self.samples = buffer_from_example(example, (batch_T, optim_B), share_memory=self.async_)
        field_names = [f for f in example._fields if f != "prev_rnn_state"]
        global SamplesToBuffer
        self.SamplesToBuffer = namedarraytuple("SamplesToBuffer", field_names)
        buffer_example = self.SamplesToBuffer(*(v for k, v in example.items()
                                                if k != "prev_rnn_state"))
        self.buffer_size = optim_B * T_target
        # self.buffer_size = sampler_B * (T_target * optim_B // sampler_B)
        self.samples = buffer_from_example(buffer_example, (batch_T, self.buffer_size), share_memory=self.async_)
        self.samples_prev_rnn_state = buffer_from_example(example.prev_rnn_state, (self.buffer_size,),
                                                          share_memory=self.async_)
        self.sleep_length = 0.01
        self.T_target = T_target
        self.t = 0
        self.optim_batch_B = optim_B

    def append_samples(self, samples):
        with self.rw_lock.write_lock:
            self._async_pull()  # Updates from other writers.
            T, B = get_leading_dims(samples, n_dim=2)  # samples.env.reward.shape[:2]
            num_new_sequences = B
            if self.t + num_new_sequences >= self.buffer_size:
                num_new_sequences = self.buffer_size - self.t
            B_idxs = np.arange(self.t, self.t + num_new_sequences)
            self.samples_prev_rnn_state[B_idxs] = samples.prev_rnn_state[0, :num_new_sequences]
            self.samples[:, self.t:self.t + num_new_sequences] = self.SamplesToBuffer(
                *(v[:, :num_new_sequences] for k, v in samples.items()
                  if k != "prev_rnn_state"))
            self._buffer_full = self._buffer_full or (self.t + num_new_sequences) == self.buffer_size
            self.t = (self.t + num_new_sequences) % self.buffer_size
            self._async_push()  # Updates to other writers + readers.

    def batch_generator(self, replay_ratio):
        if replay_ratio == 1:
            self.clear_buffer() # clear buffer so that new samples are guaranteed to be from new model parameters
            yield from self._generate_deterministic_batches()
        else:
            yield from self._generate_stochastic_minibatches(replay_ratio)

    def _generate_deterministic_batches(self):
        # replay ratioo of 1 with deterministic batch selection
        cum_sleep_length = 0
        for i in range(self.T_target):
            while True:
                with self.rw_lock:  # get read lock
                    self._async_pull()
                if self.t >= self.optim_batch_B * (i + 1) or self._buffer_full:
                    break
                time.sleep(self.sleep_length)
                cum_sleep_length += self.sleep_length if i > 0 else 0

            # batch is available
            indexes = np.arange(i * self.optim_batch_B, (i + 1) * self.optim_batch_B)
            with self.rw_lock:  # Read lock.
                batch = self.samples[:, indexes]
            yield torchify_buffer(batch), torchify_buffer(self.samples_prev_rnn_state[indexes]), cum_sleep_length

    def _generate_stochastic_minibatches(self, replay_ratio):
        cum_sleep_length = 0
        with self.rw_lock:
            self._async_pull()
        if not self._buffer_full:
            print('buffer not yet filled')
            return

        for minibatch in range(self.T_target):
            indexes = np.random.choice(self.buffer_size, self.optim_batch_B)
            with self.rw_lock:  # Read lock.
                batch = self.samples[:, indexes]

            yield torchify_buffer(batch), torchify_buffer(self.samples_prev_rnn_state[indexes]), cum_sleep_length

    def clear_buffer(self):
        with self.rw_lock.write_lock:
            self._async_pull()  # Updates from other writers.
            self._buffer_full = False
            print(f'clearing buffer from {self.t}')
            self.t = 0
            self._async_push()  # Updates to other writers + readers.
