from learning_to_be_taught.environments.meta_world.meta_world import MetaWorld
import time
import numpy as np
import gym

from learning_to_be_taught.environments.meta_world.scripted_policies import policies

class EasyReacher(MetaWorld):
    def __init__(self, demonstrations_flag=True, use_fake_demonstration=True):
        super().__init__(demonstrations_flag=demonstrations_flag, benchmark='ml1', action_repeat=15)
        self.obs_length = 6
        self.use_fake_demonstrations = use_fake_demonstration
        if use_fake_demonstration:
            self.max_demonstration_length = 1
            self.fake_demonstration = np.zeros(1)
        else:
            self.fake_demonstration = None

        if self.demonstrations_flag:
            self.observation_space = gym.spaces.Dict({
                'state': gym.spaces.Box(low=0, high=1, shape=(self.obs_length,)),
                'demonstration': gym.spaces.Box(low=0, high=1,
                                                shape=(self.max_demonstration_length, self.obs_length)),
                'demonstration_actions': self.action_space,
                'demonstration_length': gym.spaces.Discrete(200),
            })
        else:
            self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.obs_length,))
        self.env_name, self.env, self.task = self.benchmark.sample_env_and_task()
        self.env.set_task(self.task)

    def reset(self):
        # env_name, self.env, task = self.benchmark.sample_env_and_task()
        # self.env.set_task(task)
        self.env._partially_observable = False
        self.oracle_policy = policies[self.env_name]()
        cache_key = self.env_name + str(self.task.data)
        if self.demonstrations_flag:
            if cache_key not in self.demonstration_cache:
                # print('demonstration not found')
                demonstration, actions, demonstration_length, success = self.generate_demonstration(self.env, self.env_name)
                if not success:
                    print('demonstration failed')
                    return self.reset()  # try again
                self.demonstration_cache[cache_key] = (demonstration, actions, demonstration_length)

            self.demonstration_observations, self.demonstration_actions, self.demonstration_length = self.demonstration_cache[cache_key]
        obs= self.get_observation(self.env.reset())
        self.step_in_episode = 0
        self.episode_success = False
        if self.demonstrations_flag:
            obs['state'] = np.array(list(obs['state'][:3]) + list(obs['state'][9:]))
        else:
            obs = np.array(list(obs[:3]) + list(obs[9:]))
        return obs

    def step(self, action, full_obs=True):
        obs, reward, done, info = super().step(action, full_obs)
        if full_obs:
            if self.demonstrations_flag:
                obs['state'] = np.array(list(obs['state'][:3]) + list(obs['state'][9:]))
                obs['demonstration'] = self.fake_demonstration or obs['demonstration']
            else:
                obs = np.array(list(obs[:3]) + list(obs[9:]))
        return obs, reward, done, info

    def generate_demonstration(self, env, env_name):
        observations = []
        actions = []
        obs = env.reset()
        oracle_policy = policies[env_name]()
        success = False
        self.step_in_episode = 0
        demonstration_length = self.max_demonstration_length
        for step in range(self.max_demonstration_length):
            action = oracle_policy.get_action(obs)
            obs, reward, done, info = self.step(action, full_obs=False)
            success = success or info['success']
            if success or self.use_fake_demonstrations:
                success = True
                demonstration_length = min(demonstration_length, step)
            observations.append(np.array(list(obs[:3]) + list(obs[9:])))
            actions.append(action)

        return np.stack(observations), np.stack(actions), demonstration_length, success



if __name__ == '__main__':
    env = EasyReacher(demonstrations_flag=True)
    while True:
        done = False
        obs = env.reset()
        reward_sum = 0
        while not done:
            action = env.action_space.sample()
            # action = list(obs['state']) + [0,]
            action = obs['demonstration_actions']
            obs, reward, done, info = env.step(action)
            reward_sum += reward
            env.render()
            # time.sleep(0.1)
        print('reward sum: ' + str(reward_sum))

