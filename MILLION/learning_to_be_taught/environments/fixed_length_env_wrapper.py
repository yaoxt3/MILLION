from gym import Wrapper


class FixedLengthEnvWrapper(Wrapper):
    """Gym-style environment wrapper to infill the `env_info` dict of every
    ``step()`` with a pre-defined set of examples, so that `env_info` has
    those fields at every step and they are made available to the algorithm in
    the sampler's batch of data.
    """

    def __init__(self, env, fixed_episode_length=None):
        super().__init__(env)
        # self._sometimes_info = sometimes_info(**sometimes_info_kwargs)
        self.length = fixed_episode_length

    def reset(self):
        obs = super().reset()
        self.step_in_episode = 0
        self.episode_active = True
        self.fake_obs = obs
        return obs

    def step(self, action):
        """If need be, put extra fields into the `env_info` dict returned.
        See file for function ``infill_info()`` for details."""
        self.step_in_episode += 1
        o, r, d, info = super().step(action)
        if self.length is not None:
            if d:
                self.episode_active = False
                d = False
                super().reset()
            if self.step_in_episode >= self.length:
                d = True
        r = 0 if not self.episode_active else r
        return o, r, d, info

