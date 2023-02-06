import time
import torch
from rlpyt.utils.collections import namedarraytuple, NamedArrayTuple


class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, replay_buffer, batch_size): # , **sample_kwargs):
        self.replay_buffer = replay_buffer
        # self.sample_kwargs = sample_kwargs
        self.batch_size=batch_size

    def __len__(self):
        return 1000000000

    def __getitem__(self, idx):
        # return dill.dumps(self.replay_buffer.sample_batch(4))
        batch = self.replay_buffer.sample_batch(self.batch_size)._asdict()
        obs_keys = batch['all_observation']._asdict().keys()
        obs_values = batch['all_observation']._asdict().values()

        all_observation = NamedArrayTuple('obs', list(obs_keys), list(obs_values))
        batch['all_observation'] = all_observation

        batch = NamedArrayTuple('SamplesFromReplay', list(batch.keys()), list(batch.values()))
        return batch

