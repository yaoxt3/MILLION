import numpy as np
import gym
import time
import matplotlib.pyplot as plt
from random import randint
from rlpyt.envs.base import Env
from rlpyt.spaces.int_box import IntBox
from rlpyt.utils.collections import namedtuple

EnvInfo = namedtuple("EnvInfo", ["traj_done"])


class GridEnv(Env):
    def __init__(self, side_length=4, render=False):
        self.side_length = side_length
        self.pos = None
        self._observation_space = gym.spaces.Box(low=0, high=1, shape=(self.side_length * self.side_length,))
        # self._action_space = gym.spaces.Discrete(4)
        self._action_space = IntBox(low=0, high=4)
        self.goal_pos = (0, 0)
        self.render = render
        self.step_in_episode = None
        if render:
            plt.ion()
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111)
            self.plt_image = self.ax.imshow(np.ones((side_length, side_length)), cmap='gray', vmin=0, vmax=1)

    def reset(self):
        self.goal_pos = (randint(0, self.side_length - 1), randint(0, self.side_length - 1))
        self.pos = (randint(0, self.side_length - 1), randint(0, self.side_length - 1))
        # self.pos = (1, 1)
        obs = self.generate_obs()
        self.step_in_episode = 0
        return obs

    def step(self, action):
        if action == 0:
            self.pos = (max(self.pos[0] - 1, 0), self.pos[1])
        elif action == 1:
            self.pos = (min(self.pos[0] + 1, self.side_length - 1), self.pos[1])
        elif action == 2:
            self.pos = (self.pos[0], max(self.pos[1] - 1, 0))
        elif action == 3:
            self.pos = (self.pos[0], min(self.pos[1] + 1, self.side_length - 1))

        obs = self.generate_obs()
        self.step_in_episode += 1
        done = self.pos == self.goal_pos
        reward = int(done) - 0.1
        if self.step_in_episode > 8 * self.side_length:
            done = True
        info = EnvInfo(traj_done=done)
        return obs, reward, done, info

    def get_goal_obs(self):
        grid = np.zeros((self.side_length, self.side_length))
        grid[self.goal_pos] = 1
        return grid

    def generate_obs(self):
        grid = np.zeros((self.side_length, self.side_length))
        grid[self.pos] = 1
        if self.render:
            image = grid.copy()
            image[self.goal_pos] = 0.4
            self.plt_image.set_array(image)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        return grid.flatten()


class GridEnvDemonstrations(gym.Env):
    def __init__(self, side_length=4, render=False, behavioral_cloning=False, warmup_steps=0):
        self.side_length = side_length
        self.pos = None
        self.warmup_steps = warmup_steps
        self.behavioral_cloning = behavioral_cloning
        self.max_demonstration_length = side_length * 2
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Dict({
            'state': gym.spaces.Box(low=0, high=1, shape=(self.side_length * self.side_length,)),
            'image': gym.spaces.Box(low=0, high=1, shape=(self.side_length, self.side_length)),
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
        self.step_in_episode += 1
        if action == 0:
            new_pos = (max(self.pos[0] - 1, 0), self.pos[1])
        elif action == 1:
            new_pos = (min(self.pos[0] + 1, self.side_length - 1), self.pos[1])
        elif action == 2:
            new_pos = (self.pos[0], max(self.pos[1] - 1, 0))
        elif action == 3:
            new_pos = (self.pos[0], min(self.pos[1] + 1, self.side_length - 1))
        if self.step_in_episode > self.warmup_steps:
            self.pos = new_pos
            done = self.pos == self.goal_pos  # and self.step_in_episode > self.side_length
            reward = int(done) - 0.05
        else:
            done = False
            reward = 0
        obs = self.generate_obs()
        if self.behavioral_cloning:
            if self.step_in_episode >= self.warmup_steps + len(self.demonstration_actions) - 1:
                done = True
        else:
            if self.step_in_episode >= 2 * self.side_length + self.warmup_steps:
                done = True

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
        return np.stack(observations), np.stack(actions), len(actions) - 1

    def generate_obs(self):
        grid = np.zeros((self.side_length, self.side_length))
        # self.pos = (randint(0, self.side_length - 1), randint(0, self.side_length - 1))
        grid[self.pos] = 1
        demonstration_action_one_hot = np.zeros(4)
        # print('len actions:'  + str(len(self.demonstration_actions)) + ' step in episdoe: '+ str(self.step_in_episode))
        if self.behavioral_cloning:
            step = max(0, self.step_in_episode - self.warmup_steps)
            demonstration_action_one_hot[self.demonstration_actions[step]] = 1
        observation = {
            'image': grid,
            'state': grid.flatten(),
            'demonstration': self.demonstration,
            'demonstration_actions': demonstration_action_one_hot,
            'demonstration_length': np.array(self.demonstration_length)
        }
        if self.render_flag:
            image = grid.copy()
            image[self.goal_pos] = 0.4
            self.plt_image.set_array(image)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        return observation


class ComplexGridEnv(gym.Env):
    def __init__(self, side_length=5, render=False, behavioral_cloning=False, warmup_steps=0):
        self.side_length = side_length
        self.pos = None
        self.warmup_steps = warmup_steps
        self.behavioral_cloning = behavioral_cloning
        self.max_demonstration_length = self.side_length * 4
        self.observation_space = gym.spaces.Dict({
            'state': gym.spaces.Box(low=0, high=1, shape=(self.side_length * self.side_length,)),
            'image': gym.spaces.Box(low=0, high=1, shape=(self.side_length, self.side_length)),
            'demonstration': gym.spaces.Box(low=0, high=1,
                                            shape=(self.max_demonstration_length, self.side_length * self.side_length)),
            'demonstration_actions': gym.spaces.Box(low=0, high=1, shape=(4,)),
            'demonstration_length': gym.spaces.Discrete(100)
        })
        self.action_space = gym.spaces.Discrete(4)
        self.goal_pos = (0, 0)
        self.render_flag = render
        self.step_in_episode = None
        self.box_pos = None
        self.demonstration = self.demonstration_actions = None
        if render:
            plt.ion()
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111)
            self.plt_image = self.ax.imshow(np.ones((side_length, side_length)), cmap='gray', vmin=0, vmax=1)

    def reset(self):
        self.goal_pos = (randint(0, self.side_length - 1), randint(0, self.side_length - 1))
        self.box_pos = (randint(1, self.side_length - 2), randint(1, self.side_length - 2))  # don't start at border
        self.pos = (randint(0, self.side_length - 1), randint(0, self.side_length - 1))
        if self.pos == self.box_pos or self.box_pos == self.goal_pos:
            return self.reset()  # call recursively until valid start positions are found
        self.step_in_episode = 0
        self.demonstration, self.demonstration_actions, self.demonstration_length = self.get_demonstration()
        self.step_in_episode = 0
        obs = self.generate_obs()
        return obs

    def step(self, action, full_obs=True):
        self.step_in_episode += 1
        if self.step_in_episode > self.warmup_steps or not full_obs:
            self.pos, self.box_pos = self.player_move(action)
            done = self.box_pos == self.goal_pos
            reward = int(done) - 0.05
        else:
            done = False
            reward = 0
        obs = self.generate_obs(full_obs=full_obs)
        if self.behavioral_cloning:
            if full_obs and self.step_in_episode >= self.warmup_steps + len(self.demonstration_actions) - 1:
                done = True
        else:
            if self.step_in_episode >= 3 * self.side_length + self.warmup_steps:
                done = True
        return obs, reward, done, {}

    def player_move(self, action):
        """
        computes new player position and box position
        :param action: env action
        :return: new_player_pos, new_box_pos
        """
        old_player_pos = self.pos
        old_box_pos = self.box_pos
        if action == 0:
            new_player_pos = (max(self.pos[0] - 1, 0), self.pos[1])
        elif action == 1:
            new_player_pos = (min(self.pos[0] + 1, self.side_length - 1), self.pos[1])
        elif action == 2:
            new_player_pos = (self.pos[0], max(self.pos[1] - 1, 0))
        elif action == 3:
            new_player_pos = (self.pos[0], min(self.pos[1] + 1, self.side_length - 1))
        else:
            raise BaseException('action invalid')

        if new_player_pos == old_box_pos:
            # box was moved
            new_box_pos = tuple(old_box_pos + (np.array(new_player_pos) - np.array(old_player_pos)))
            if new_box_pos[0] < 0 or new_box_pos[0] >= self.side_length or new_box_pos[1] < 0 or new_box_pos[
                1] >= self.side_length:
                # Box can't be moved
                new_player_pos, new_box_pos = old_player_pos, old_box_pos
        else:
            new_box_pos = old_box_pos

        return new_player_pos, new_box_pos

    def get_goal_obs(self):
        grid = np.zeros((self.side_length, self.side_length))
        grid[self.goal_pos] = 1
        return grid

    def get_demonstration(self):
        agent_start_pos, box_start_pos = self.pos, self.box_pos  # save episdoe settings
        done = False
        observations = []
        actions = []
        while not done:
            action = self.oracle()
            print('oracle action: ' + str(action))
            obs, _, done, _ = self.step(action, full_obs=False)
            observations.append(obs)
            actions.append(action)

        demonstration_length = len(observations)
        # fill up to demonstration length
        for t in range(len(observations), self.max_demonstration_length):
            observations.append(observations[-1])
            actions.append(actions[-1])

        self.pos, self.box_pos = agent_start_pos, box_start_pos  # restore episode start
        return np.stack(observations), np.stack(actions), demonstration_length

    def generate_obs(self, full_obs=True):
        grid = np.zeros((self.side_length, self.side_length))
        grid[self.pos] = 1
        grid[self.box_pos] = 0.5
        if full_obs:
            demonstration_action_one_hot = np.zeros(4)
            if self.behavioral_cloning:
                step = max(0, self.step_in_episode - self.warmup_steps)
                print('returning action at step: ' + str(step))
                demonstration_action_one_hot[self.demonstration_actions[step]] = 1
            observation = {
                'image': grid,
                'state': grid.flatten(),
                'demonstration': self.demonstration,
                'demonstration_actions': demonstration_action_one_hot,
                'demonstration_length': np.array(self.demonstration_length)
            }
            if self.render_flag:
                image = grid.copy()
                image[self.goal_pos] = 0.2
                self.plt_image.set_array(image)
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
        else:
            observation = grid.flatten()
        return observation

    def direction_to_action(self, direction: list):
        if direction == [-1, 0]:
            action = 0
        elif direction == [1, 0]:
            action = 1
        elif direction == [0, -1]:
            action = 2
        elif direction == [0, 1]:
            action = 3
        else:
            raise Exception('direction: ' + str(direction) + ' is invalid')

        return action

    def oracle(self):
        """
        :return: ideal action in current state
        """
        agent_pos = np.array(self.pos)
        box_pos = np.array(self.box_pos)
        if self.box_pos[0] < self.goal_pos[0]:
            push_direction = [1, 0]
        elif self.box_pos[0] > self.goal_pos[0]:
            push_direction = [-1, 0]
        elif self.box_pos[1] < self.goal_pos[1]:
            push_direction = [0, 1]
        elif self.box_pos[1] > self.goal_pos[1]:
            push_direction = [0, -1]

        push_direction = np.array(push_direction)
        if max([np.equal(agent_pos + x * push_direction, box_pos).all() for x in range(1, self.side_length)]):
            # box in front
            action = self.direction_to_action(list(push_direction))
        elif max([np.equal(agent_pos - x * push_direction, box_pos).all() for x in range(1, self.side_length)]):
            # box behind -> step to side
            action = self.direction_to_action(list(reversed(push_direction)))
        else:
            side_direction = (box_pos - agent_pos) * np.abs(np.flip(push_direction))
            side_step = np.clip(side_direction, a_min=-1, a_max=1)
            possible_side_pos = agent_pos + side_direction
            if max([np.equal(possible_side_pos + x * push_direction, box_pos).all() for x in
                    range(1, self.side_length)]):
                action = self.direction_to_action(list(side_step))
            else:
                action = self.direction_to_action(list(-push_direction))

        return action


class GridEnvCombination(gym.Env):
    def __init__(self, side_length=4, render=False):
        self.envs = [GridEnvDemonstrations(side_length, render), ComplexGridEnv(side_length, render)]
        self.env_index = None
        self.action_space = self.envs[0].action_space
        self.observation_space = self.envs[0].observation_space

    def reset(self):
        self.env_index = randint(0, len(self.envs) - 1)
        return self.envs[self.env_index].reset()

    def step(self, action):
        return self.envs[self.env_index].step(action)

    def render(self, mode='human'):
        return self.envs[self.env_index].render(mode)


class GridEnvOracle:
    @staticmethod
    def get_action(grid_env_object):
        if grid_env_object.pos[0] > grid_env_object.goal_pos[0]:
            return 0
        if grid_env_object.pos[0] < grid_env_object.goal_pos[0]:
            return 1
        elif grid_env_object.pos[1] > grid_env_object.goal_pos[1]:
            return 2
        else:
            return 3

    @staticmethod
    def get_goal_obs(grid_env_object):
        grid = np.zeros((grid_env_object.side_length, grid_env_object.side_length))
        grid[grid_env_object.goal_pos] = 1
        return grid


if __name__ == '__main__':
    side_length = 4
    # env = GridEnv(side_length=side_length, render=True)
    # env = GridEnvDemonstrations(side_length=side_length, render=True, behavioral_cloning=True, warmup_steps=2)
    env = ComplexGridEnv(side_length=side_length, render=True, behavioral_cloning=True, warmup_steps=3)
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
            # print(obs['demonstration_length'])
            action = np.argmax(obs['demonstration_actions'])
            # print('action: ' + str(action))
            # action = env.oracle()
            # action = GridEnvOracle.get_action(env)
            # print('action: ' + str(action))
            start = time.time()
            obs, reward, done, info = env.step(action)
            print('reward: '+ str(reward))
            # print('step took: ' + str(time.time() - start))
            reward_sum += reward
            time.sleep(0.12)
            step += 1
        # print('reward sum: ' + str(reward_sum))
