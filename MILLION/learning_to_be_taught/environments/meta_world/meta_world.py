import gym
import numpy as np
import time
from learning_to_be_taught.environments.meta_world.scripted_policies import policies
from learning_to_be_taught.environments.meta_world.meta_world_benchmarks import ML10, ML1, ML45, SingleEnv


class MetaWorld(gym.Env):
    def __init__(self, demonstrations_flag=True, benchmark='ml10', action_repeat=2, time_observation=False, **kwargs):
        if benchmark == 'ml1':
            self.benchmark = ML1(**kwargs)
        elif benchmark == 'ml10':
            self.benchmark = ML10(**kwargs)
        elif benchmark == 'ml45':
            self.benchmark = ML45(**kwargs)
        else:
            self.benchmark = SingleEnv(env_name=benchmark, **kwargs)

        self.action_space = gym.spaces.Box(low=np.array([-1, -1, -1, -1]), high=np.array([1, 1, 1, 1]))
        self.reward_scale = 1
        self.step_in_episode = None
        self.action_repeat = action_repeat
        self.demonstration_action_repeat = 3
        self.max_demonstration_length = 150 // self.action_repeat
        self.time_observation = time_observation
        self.state_length = 13 if time_observation else 12
        # self.envs = [env_cls() for env_cls in self.benchmark.train_classes]
        self.demonstrations_flag = demonstrations_flag
        if self.demonstrations_flag:
            self.observation_space = gym.spaces.Dict({
                'state': gym.spaces.Box(low=-1, high=1, shape=(self.state_length,)),
                'demonstration': gym.spaces.Box(low=-1, high=1,
                                                shape=(self.max_demonstration_length, self.state_length)),
                'demonstration_actions': self.action_space,
                'demonstration_length': gym.spaces.Discrete(200),
                'task_id': gym.spaces.Discrete(50)
            })
        else:
            self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(self.state_length,))
        self.demonstration_cache = dict()
        self.bad_tasks = []
        self.demonstration_actions = self.demonstration_observations = self.oracle_policy = None
        self.episode_success = None

    def reset(self):
        self.env_name, self.env, task, self.env_id = self.benchmark.sample_env_and_task()
        task_key = str(task.data)
        if task_key in self.bad_tasks:
            return self.reset()
        # print(f'env name {self.env_name} super classes: {self.benchmark._train_classes}')
        self.env.set_task(task)
        self.env._partially_observable = False
        self.oracle_policy = policies[self.env_name]()
        cache_key = self.env_name + str(task.data)
        if self.demonstrations_flag:
            if cache_key not in self.demonstration_cache:
                # print('demonstration not found')
                demonstration, actions, demonstration_length, success = self.generate_demonstration(self.env, self.env_name)
                if not success:
                    print('demonstration failed')
                    self.bad_tasks.append(task_key)
                    return self.reset()  # try again
                self.demonstration_cache[cache_key] = (demonstration, actions, demonstration_length)

            self.demonstration_observations, self.demonstration_actions, self.demonstration_length = self.demonstration_cache[cache_key]

        self.step_in_episode = 0
        observation = self.get_observation(self.env.reset())
        self.episode_success = False
        return observation

    def generate_demonstration(self, env, env_name):
        observations = []
        actions = []
        obs = env.reset()
        self.step_in_episode = 0
        oracle_policy = policies[env_name]()
        success = False
        demonstration_length = self.max_demonstration_length
        for step in range(self.max_demonstration_length):
            # obs = np.concatenate((obs, [self.step_in_episode * self.action_repeat / 150]))
            action = oracle_policy.get_action(obs) #Jk[:-self.time_observation])
            obs, reward, done, info = self.step(action, full_obs=False)
            success = success or info['success']
            if success:
                demonstration_length = min(demonstration_length, step)
            if self.time_observation:
                observations.append(np.concatenate((obs, [self.step_in_episode * self.action_repeat / 150])))
            else:
                observations.append(obs)
            actions.append(action)
        observations = np.stack(observations)[::self.demonstration_action_repeat]
        return np.stack(observations), np.stack(actions), demonstration_length, success

    def step(self, action, full_obs=True):
        self.step_in_episode += 1
        reward_sum = 0
        for i in range(self.action_repeat):
            obs, reward, done, info = self.env.step(action)
            reward_sum += reward
        # if (self.step_in_episode + 1) * self.action_repeat >= self.env.max_path_length or info['success']:
        # obs = np.concatenate((obs, [self.step_in_episode * self.action_repeat / 150]))
        if (self.step_in_episode + 1) * self.action_repeat > 150: # fix max path length to 150
            done = True
        if info['success']:
            self.episode_success = True
        info['episode_success'] = self.episode_success
        # info[self.env_name + '_episode_success'] = self.episode_success
        info = self.append_env_info(info)

        if full_obs:
            obs = self.get_observation(current_state=obs)

        return obs, reward_sum, done, info

    def get_observation(self, current_state):
        demonstration_action = self.oracle_policy.get_action(current_state)
        if self.time_observation:
            current_state = np.concatenate((current_state, [self.step_in_episode * self.action_repeat / 150]))

        if self.demonstrations_flag:
            observation = {
                'state': current_state,
                'demonstration': self.demonstration_observations,
                'demonstration_actions': demonstration_action,
                'demonstration_length': np.array(self.demonstration_length),
                'task_id': np.array(self.env_id)
            }
        else:
            return current_state
        return observation

    def render(self, *args, **kwargs):
        self.env.render(*args, **kwargs)

    def append_env_info(self, info):
        for env_name in self.benchmark.all_possible_classes.keys():
            info[env_name.replace('-', '') + '_episode_success'] = float('nan')

        info[self.env_name.replace('-', '') + '_episode_success'] = self.episode_success
        return info


if __name__ == '__main__':
    s = time.time()
    env = MetaWorld(benchmark='ml10', action_repeat=3, demonstrations_flag=True, time_observation=False, sample_num_classes=1)
    print('env init took: '+ str(time.time() - s))
    while True:
        s = time.time()
        obs = env.reset()  # Reset environment
        # print('reset took: ' + str(time.time() - s))
        done = False
        step = 0
        reward_sum= 0
        while not done:
            action = env.action_space.sample()  # Sample an action
            # print([obs['state'][-1]])
            print(f'task {obs["task_id"]}')
            # action = env.oracle_policy.get_action(obs['state'])
            # action = obs[9:]
            # action = env.oracle_policy.get_action(obs)
            # a *= 0
            s = time.time()
            # print(np.isnan(obs.any()))
            obs, reward, done, info = env.step(action)  # Step the environoment with the sampled random action
            print(f'step took : {time.time() - s}')
            # print('step toook: ' + str(time.time() - s))
            # print('reward: ' + str(reward))
            reward_sum += reward
            env.render(mode='human')
            step += 1
            time.sleep(0.004)

        # print('num steps: ' + str(step) + ' success: ' + str(info['episode_success']))
        # print('reward sum: ' + str(reward_sum))
