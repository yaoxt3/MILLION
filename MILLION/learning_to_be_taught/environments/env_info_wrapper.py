
from gym import Wrapper



class EnvInfoWrapper(Wrapper):
    """Gym-style environment wrapper to infill the `env_info` dict of every
    ``step()`` with a pre-defined set of examples, so that `env_info` has
    those fields at every step and they are made available to the algorithm in
    the sampler's batch of data.
    """

    def __init__(self, env, info_example):
        super().__init__(env)
        # self._sometimes_info = sometimes_info(**sometimes_info_kwargs)
        self._sometimes_info = info_example

    def step(self, action):
        """If need be, put extra fields into the `env_info` dict returned.
        See file for function ``infill_info()`` for details."""
        o, r, d, info = super().step(action)
        for k, v in info.items():
            if v is None:
                info[k] = 0
        # Try to make info dict same key structure at every step.
        return o, r, d, infill_info(info, self._sometimes_info)

def infill_info(info, sometimes_info):
    for k, v in sometimes_info.items():
        if k not in info:
            info[k] = v
        elif isinstance(v, dict):
            infill_info(info[k], v)
    return info
