import numpy as np
import gym


class VectorizeObs:
    def __init__(self, env):
        self._env = env
        self.observation_space = gym.spaces.Box(low=env.observation_space.low.flatten(),
                                                high=env.observation_space.high.flatten())
        self.obs = None

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        self.obs = self._vectorize_obs(obs)
        return self.obs, reward, done, info

    def reset(self):
        self.obs = self._vectorize_obs(self._env.reset())
        return self.obs

    def render(self, *args, **kwargs):
        self._env.render_flag(*args, **kwargs)

    def get_goal_obs(self):
        return self._vectorize_obs(self._env.get_goal_obs())

    def _vectorize_obs(self, obs_2d):
        return obs_2d.flatten()

    def get_image_obs(self):
        return self.obs.reshape((self._env.side_length, self._env.side_length))


class TimeLimit:

    def __init__(self, env, duration):
        self._env = env
        self._duration = duration
        self._step = None

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        assert self._step is not None, 'Must reset environment.'
        obs, reward, done, info = self._env.step(action)
        self._step += 1
        if self._step >= self._duration:
            done = True
            if 'discount' not in info:
                info['discount'] = np.array(1.0).astype(np.float32)
            self._step = None
        return obs, reward, done, info

    def reset(self):
        self._step = 0
        return self._env.reset()


class ConcatGoalObs:
    def __init__(self, env):
        self._env = env
        self.observation_space = gym.spaces.Box(low=env.observation_space.low.repeat(2),
                                                high=env.observation_space.high.repeat(2))

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        return self._concat_goal_obs(obs), reward, done, info

    def reset(self):
        return self._concat_goal_obs(self._env.reset())

    def _concat_goal_obs(self, obs):
        return np.concatenate((obs, self._env.get_goal_obs()))


class RewardObs:

    def __init__(self, env):
        self._env = env
        obs_space = env.observation_space
        assert len(obs_space.shape) == 1, 'obs space has to be 1D for Reward obs wrapper'
        self.observation_space = gym.spaces.Box(np.concatenate((obs_space.low, 0)), np.concatenate((obs_space, 1)))

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        obs = np.concatenate((obs, reward))
        return obs, reward, done, info

    def reset(self):
        obs = self._env.reset()
        np.concatenate((obs, 0))
        return obs


# class DemonstrationObs:
#     def __init__(self, env, demonstrator):
#         self._env = env
#         self.observation_space = gym.spaces.Dict({
#             'obs': env.observation_space
#             'demonstration', gym.spaces.Box(low=np.repeat(env.observation_space, ))
#         })
#
#     def __getattr__(self, name):
#         return getattr(self._env, name)
#
#     def step(self, action):
#         obs, reward, done, info = self._env.step(action)
#         return self._concat_goal_obs(obs), reward, done, info
#
#     def reset(self):
#         return self._concat_goal_obs(self._env.reset())
#
#     def _concat_goal_obs(self, obs):
#         return np.concatenate((obs, self._env.get_goal_obs()))
#
