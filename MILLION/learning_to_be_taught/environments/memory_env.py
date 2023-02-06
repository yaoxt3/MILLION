import gym
import numpy as np
import time
from learning_to_be_taught.environments.meta_world.scripted_policies import policies
from collections import deque


class MemoryEnv(gym.Env):
    def __init__(self, delay=10, action_size=1):
        self.action_space = gym.spaces.Box(shape=(action_size,), low=-1, high=1)
        self.step_in_episode = None
        self.delay = delay
        self.action_size = action_size
        self.observation_space = gym.spaces.Dict({
            'state': gym.spaces.Box(low=-1, high=1, shape=(action_size,)),
            'task_id': gym.spaces.Discrete(50)
        })
        self.oracle_policy = None

    def reset(self):
        self.step_in_episode = 0
        self.value = [np.random.uniform(-1, 1, (self.action_size,)) for _ in range(self.delay)]
        return dict(state=self.value[-1], task_id=np.array(0))

    def step(self, action):
        self.step_in_episode += 1
        if self.step_in_episode >= self.delay:
            reward = -float(np.square(action - self.value[0]))
        else:
            reward = 0

        obs = np.random.uniform(-1, 1, (self.action_size,))
        self.value = self.value[1:] + [obs]

        full_obs = dict(
            state=obs,
            task_id=np.array(0)
        )
        done = self.step_in_episode >= 40 * 5

        return full_obs, reward, done, {}

    def render(self, *args, **kwargs):
        pass



if __name__ == '__main__':
    s = time.time()
    env = MemoryEnv(delay=1)
    while True:
        obs = env.reset()  # Reset environment
        done = False
        step = 0
        reward_sum= 0
        while not done:
            action = env.action_space.sample()  # Sample an action
            s = time.time()
            action = obs['state']
            obs, reward, done, info = env.step(action)  # Step the environoment with the sampled random action
            # print(f'step took : {time.time() - s}')
            # print(obs)
            # print('step toook: ' + str(time.time() - s))
            print('reward: ' + str(reward))
            reward_sum += reward
            env.render(mode='human')
            step += 1
            time.sleep(0.04)

        print('num steps: ' + str(step))
        # print('reward sum: ' + str(reward_sum))
        print(f'info {info}')
