import gym
import numpy as np
from learning_to_be_taught.environments.classic_control.acrobot_backend import AcrobotEnv


class Acrobot(gym.Env):
    def __init__(self, demonstrations_flag=True, max_episode_length=50):
        self.action_repeat = 5
        self.step_in_episode = None
        self.max_episode_length = max_episode_length
        self.max_demonstration_length = max_episode_length // self.action_repeat
        self.demonstration_length = 1
        self.reward_scale = 1
        self.demonstrations_flag = demonstrations_flag
        self.env = AcrobotEnv()
        self.action_space = self.env.action_space
        if self.demonstrations_flag:
            demonstration_low = self.env.observation_space.low.repeat(self.max_demonstration_length).reshape(
                self.max_demonstration_length, -1)
            demonstration_high = self.env.observation_space.high.repeat(self.max_demonstration_length).reshape(
                self.max_demonstration_length, -1)

            self.observation_space = gym.spaces.Dict({
                'state': self.env.observation_space,
                'demonstration': gym.spaces.Box(low=demonstration_low, high=demonstration_high),
                'demonstration_actions': self.action_space,
                'demonstration_length': gym.spaces.Discrete(self.max_demonstration_length),
            })
        else:
            self.observation_space = self.env.observation_space

        self.fake_demonstration = np.stack([self.env.observation_space.sample()])
        self.episode_success = None

    def reset(self):
        observation = self.get_observation(self.env.reset())
        self.step_in_episode = 0
        self.episode_success = False
        return observation

    def step(self, action, full_obs=True):
        self.step_in_episode += 1
        reward_sum = 0
        for _ in range(self.action_repeat):
            state, reward, done, info = self.env.step(action)
            reward_sum += reward * self.reward_scale

        # done = done or self.step_in_episode >= self.max_episode_length
        done = self.step_in_episode >= self.max_episode_length
        return self.get_observation(state), reward_sum/self.action_repeat, done, info

    def get_observation(self, current_state):
        if self.demonstrations_flag:
            observation = {
                'state': current_state,
                'demonstration': self.fake_demonstration,
                'demonstration_actions': self.action_space.sample(),
                'demonstration_length': np.array(self.demonstration_length)
            }
        else:
            return current_state
        return observation

    def render(self, *args, **kwargs):
        self.env.render(*args, **kwargs)


if __name__ == '__main__':

    # env = gym.make('Pendulum-v0')
    env = Acrobot()
    while True:
        done = False
        obs = env.reset()
        print('reset')
        while not done:
            action = env.action_space.sample()
            # print(action)
            obs, reward, done, info = env.step(action)
            # print('reward; ' + str(reward))
            # print(obs)
            env.render()

