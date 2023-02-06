import gym
import time
from abc import abstractmethod
import matplotlib.pyplot as plt
from random import randint
import numpy as np


class DemonstrationEnvironment(gym.Env):
    def __init__(self, state_space, image_space, action_space):
        self.observation_space = gym.spaces.Dict({
            'demonstration_flag': gym.spaces.Discrete(2), # 1: demonstration observation, 0: agent is acting
            'action': action_space, # used for demonstrations with actions
            'state': state_space,
            'image': image_space
        })

        self.action_space = action_space
        self.demonstration_flag = None

    def reset(self):
        self.demonstration_flag = True
        return self._reset()

    @abstractmethod
    def _reset(self):
        pass

    def step(self):
        if self.demonstration_flag:
            self._step_demonstration()


class EfficientGridEnv(gym.Env):
    def __init__(self, side_length=4, render=False, behavioral_cloning=False):
        self.side_length = side_length
        self.pos = None
        self.behavioral_cloning = behavioral_cloning
        self.demonstration_length = side_length * 2
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Dict({
            'state': gym.spaces.Box(low=0, high=1, shape=(self.side_length * self.side_length,)),
            'oracle_action': self.action_space,
            'demonstration_flag': gym.spaces.Discrete(1)
        })
        self.goal_pos = (0, 0)
        self.render_flag = render
        self.step_in_episode = None
        self.demonstration_flag = None
        self.max_episode_length = 2 * self.side_length
        if render:
            plt.ion()
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111)
            self.plt_image = self.ax.imshow(np.ones((side_length, side_length)), cmap='gray', vmin=0, vmax=1)

    def reset(self):
        self.goal_pos = (randint(0, self.side_length - 1), randint(0, self.side_length - 1))
        self.start_pos = (randint(0, self.side_length - 1), randint(0, self.side_length - 1))
        self.pos = self.start_pos
        if self.goal_pos == self.pos:
            return self.reset()
        self.step_in_episode = 0
        self.demonstration_flag = True
        obs = self.generate_obs()
        return obs

    def step(self, action):
        if self.demonstration_flag:
            action = self.oracle()
        if action == 0:
            self.pos = (max(self.pos[0] - 1, 0), self.pos[1])
        elif action == 1:
            self.pos = (min(self.pos[0] + 1, self.side_length - 1), self.pos[1])
        elif action == 2:
            self.pos = (self.pos[0], max(self.pos[1] - 1, 0))
        elif action == 3:
            self.pos = (self.pos[0], min(self.pos[1] + 1, self.side_length - 1))

        self.step_in_episode += 1
        reward = -0.05
        done = self.step_in_episode >= self.max_episode_length
        if self.pos == self.goal_pos:
            if self.demonstration_flag:
                self.demonstration_flag = False
                self.pos = self.start_pos
            else:
                done = True
                reward += 1

        obs = self.generate_obs()
        return obs, reward, done, {}

    def oracle(self):
        pos = list(self.pos)
        if pos[0] > self.goal_pos[0]:
            return 0
        elif pos[0] < self.goal_pos[0]:
            return 1
        elif pos[1] > self.goal_pos[1]:
            return 2
        elif pos[1] < self.goal_pos[1]:
            return 3
        else:
            return 0 # goal reached; return random action

    def get_demonstration(self):
        pos = list(self.pos)
        observations = []
        actions = []
        for step in range(self.demonstration_length):
            grid = np.zeros((self.side_length, self.side_length))
            grid[tuple(pos)] = 1
            observations.append(grid.flatten())
            if pos[0] > self.goal_pos[0]:
                pos[0] -= 1
                actions.append(0)
            elif pos[0] < self.goal_pos[0]:
                pos[0] += 1
                actions.append(1)
            elif pos[1] > self.goal_pos[1]:
                pos[1] -= 1
                actions.append(2)
            elif pos[1] < self.goal_pos[1]:
                pos[1] += 1
                actions.append(3)
            # else:
            #     actions.append(4)
        actions.append(0)  # fake action for step at episode end
        return np.stack(observations), np.stack(actions)

    def generate_obs(self):
        grid = np.zeros((self.side_length, self.side_length))
        grid[self.pos] = 1
        oracle_action_one_hot = np.zeros(4)
        oracle_action_one_hot[self.oracle()] = 1
        observation = {
            'state': grid.flatten(),
            'oracle_action': oracle_action_one_hot,
            'demonstration_flag': self.demonstration_flag
        }
        if self.render_flag:
            image = grid.copy()
            image[self.goal_pos] = 0.4
            self.plt_image.set_array(image)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        return observation

if __name__ == '__main__':
    side_length = 8
    # env = GridEnv(side_length=side_length, render=True)
    # env = GridEnvDemonstrations(side_length=side_length, render=True)
    env = EfficientGridEnv(side_length=side_length, render=True)
    # env = GridEnvCombination(side_length=side_length, render=True)

    while True:
        start = time.time()
        obs = env.reset()
        print('reset took: ' + str(time.time() - start))
        done = False
        reward_sum = 0
        step = 0
        while not done:
            action = env.action_space.sample()
            # action = np.argmax(obs['demonstration_actions'])
            start = time.time()
            obs, reward, done, info = env.step(action)
            # print('step took: ' + str(time.time() - start))
            reward_sum += reward
            time.sleep(0.02)
            step += 1
        # print('reward sum: ' + str(reward_sum))
