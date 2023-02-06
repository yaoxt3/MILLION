import gym
import time
import numpy as np
from random import randint
import matplotlib.pyplot as plt


class ActionCloningEnv(gym.Env):
    def __init__(self, side_length=4, render=False, behavioral_cloning=False):
        self.side_length = side_length
        self.pos = None
        self.behavioral_cloning = behavioral_cloning
        self.max_demonstration_length = 10
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Dict({
            'state': gym.spaces.Box(low=0, high=1, shape=(self.side_length * self.side_length,)),
            # 'image': gym.spaces.Box(low=0, high=1, shape=(self.side_length, self.side_length)),
            'demonstration': gym.spaces.Box(low=0, high=1,
                                            shape=(self.max_demonstration_length, self.side_length * self.side_length)),
            'demonstration_actions': gym.spaces.Box(low=0, high=1, shape=(4,)),
            'demonstration_length': gym.spaces.Discrete(100),
        })
        self.goal_pos = (0, 0)
        self.render_flag = render
        self.step_in_episode = None
        if render:
            plt.ion()
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111)
            self.plt_image = self.ax.imshow(np.ones((side_length, side_length)), cmap='gray', vmin=0, vmax=1)

    def reset(self):
        self.goal_pos = (randint(0, self.side_length - 1), randint(0, self.side_length - 1))
        self.pos = (randint(0, self.side_length - 1), randint(0, self.side_length - 1))
        self.step_in_episode = 0
        if self.goal_pos == self.pos:
            return self.reset()
        self.demonstration, self.demonstration_actions, self.demonstration_length = self.get_demonstration()
        obs = self.generate_obs()
        self.step_in_episode = 0
        return obs

    def step(self, action):
        if self.step_in_episode > 0:
            if action == 0:
                self.pos = (max(self.pos[0] - 1, 0), self.pos[1])
            elif action == 1:
                self.pos = (min(self.pos[0] + 1, self.side_length - 1), self.pos[1])
            elif action == 2:
                self.pos = (self.pos[0], max(self.pos[1] - 1, 0))
            elif action == 3:
                self.pos = (self.pos[0], min(self.pos[1] + 1, self.side_length - 1))

        obs = self.generate_obs()
        reward = int((obs['state'] == self.demonstration[self.step_in_episode]).all())
        done = self.step_in_episode == self.max_demonstration_length - 1
        self.step_in_episode += 1

        return obs, reward, done, {}

    def get_goal_obs(self):
        grid = np.zeros((self.side_length, self.side_length))
        grid[self.goal_pos] = 1
        return grid

    def get_demonstration(self):
        pos = list(self.pos)
        observations = []
        actions = []
        for step in range(self.max_demonstration_length):
            grid = np.zeros((self.side_length, self.side_length))
            grid[tuple(pos)] = 1
            observations.append(grid.flatten())
            action = randint(0, 3)
            actions.append(action)
            if action == 0:
                pos = (max(pos[0] - 1, 0), pos[1])
            elif action == 1:
                pos = (min(pos[0] + 1, self.side_length - 1), pos[1])
            elif action == 2:
                pos = (pos[0], max(pos[1] - 1, 0))
            elif action == 3:
                pos = (pos[0], min(pos[1] + 1, self.side_length - 1))

        actions.append(0)  # fake action for step at episode end
        return np.stack(observations), np.stack(actions), len(actions) - 1

    def generate_obs(self):
        grid = np.zeros((self.side_length, self.side_length))
        # self.pos = (randint(0, self.side_length - 1), randint(0, self.side_length - 1))
        grid[self.pos] = 1
        demonstration_action_one_hot = np.zeros(4)
        demonstration_action_one_hot[self.demonstration_actions[self.step_in_episode]] = 1
        observation = {
            # 'image': grid,
            'state': grid.flatten(),
            'demonstration': self.demonstration,
            'demonstration_actions': demonstration_action_one_hot,
            'demonstration_length': np.array(self.demonstration_length)
        }
        if self.render_flag:
            image = grid.copy()
            goal_obs = self.demonstration[self.step_in_episode].reshape(self.side_length, self.side_length)
            image[np.unravel_index(goal_obs.argmax(), goal_obs.shape)] = 0.4
            image[self.pos] = 1
            self.plt_image.set_array(image)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        return observation



if __name__ == '__main__':
    side_length = 2
    # env = GridEnv(side_length=side_length, render=True)
    env = ActionCloningEnv(side_length=side_length, render=True)
    # env = ComplexGridEnv(side_length=side_length, render=True)
    # env = GridEnvCombination(side_length=side_length, render=True)

    while True:
        obs = env.reset()
        done = False
        reward_sum = 0
        step = 0
        while not done:
            action = env.action_space.sample()
            # print(obs['demonstration_length'])
            action = np.argmax(obs['demonstration_actions'])
            # print('action: ' + str(action))
            start = time.time()
            obs, reward, done, info = env.step(action)
            # print('step took: ' + str(time.time() - start))
            reward_sum += reward
            # time.sleep(1.2)
            step += 1
        print('reward sum: ' + str(reward_sum))
