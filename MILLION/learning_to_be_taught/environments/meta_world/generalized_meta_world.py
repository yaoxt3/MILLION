import gym
import numpy as np
import time
from learning_to_be_taught.environments.meta_world.scripted_policies import policies
from learning_to_be_taught.environments.meta_world.meta_world_benchmarks import ML10, ML1, SingleEnv, ML45
# from learning_to_be_taught.environments.meta_world.meta_world_benchmarks_v2 import ML10V2, ML1V2, SingleEnvV2, ML45V2
import mujoco_py


class GeneralizedMetaWorld(gym.Env):
    def __init__(self, benchmark='ml10', action_repeat=1, demonstration_action_repeat=3, max_trials_per_episode=1,
                 dense_rewards=False, prev_action_obs=False, demonstrations=True, partially_observable=True,
                 visual_observations=False, v2=False, **kwargs):
        self.v2 = v2
        if benchmark == 'ml1':
            self.benchmark = ML1V2(**kwargs) if v2 else ML1(**kwargs)
        elif benchmark == 'ml10':
            self.benchmark = ML10V2(**kwargs) if v2 else ML10(**kwargs)
        elif benchmark == 'ml45':
            self.benchmark = ML45V2(**kwargs) if v2 else ML45(**kwargs)
        else:
            self.benchmark = SingleEnvV2(env_name=benchmark) if v2 else SingleEnv(env_name=benchmark, **kwargs)

        self.action_space = gym.spaces.Box(low=np.array([-1, -1, -1, -1]), high=np.array([1, 1, 1, 1]))
        self.step_in_episode = None
        self.action_repeat = action_repeat
        self.demonstration_action_repeat = demonstration_action_repeat
        self.max_trials_per_episode = max_trials_per_episode
        self.visual_observations = visual_observations
        obs_size = 15
        if prev_action_obs:
            obs_size += np.prod(self.action_space.shape)
        if self.visual_observations:
            self.observation_space = gym.spaces.Dict({
                'camera_image': gym.spaces.Box(low=-1, high=1, shape=(3, 64, 64)),
                'state': gym.spaces.Box(low=-10, high=10, shape=(obs_size,)),
            })
            # self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(3, 64, 64))
        else:
            self.observation_space = gym.spaces.Dict({
                'state': gym.spaces.Box(low=-10, high=10, shape=(obs_size,)),
                # 'camera_image': gym.spaces.Box(low=-10, high=10, shape=(128, 128, 3)),
                'task_id': gym.spaces.Discrete(50),
                'demonstration_phase': gym.spaces.Discrete(2),
            })
        self.dense_rewards = dense_rewards
        self.oracle_policy = None
        self.prev_action_obs = prev_action_obs
        self.demonstrations = demonstrations
        self.partially_observable = partially_observable

    def reset(self):
        self._reset_env(new_episode=True)
        self.oracle_policy = policies[self.env_name]() if not self.v2 else None
        self.trial_in_episode = 0
        self.demonstration_phase = True if self.demonstrations else False
        self.num_successful_trials = 0
        self.demonstration_successes = 0
        return self.get_full_observation(self.observation)

    def _reset_env(self, new_episode=False):
        if new_episode:
            self.env_name, self.env, task, self.env_id = self.benchmark.sample_env_and_task()
            self.env.set_task(task)
            if self.visual_observations:
                self.setup_camera()
        self.step_in_trial = 0
        self.observation = self.env.reset()
        self.trial_success = False
        self.demonstration_success = False

    def setup_camera(self):
        if not hasattr(self.env, 'viewer') or self.env.viewer is None:
            viewer = mujoco_py.MjRenderContextOffscreen(self.env.sim, 0)
            viewer.cam.distance = 1
            viewer.cam.fixedcamid = 0
            viewer.cam.azimuth = 130.0
            viewer.cam.elevation = -50
            viewer.cam.lookat[0] = 0
            viewer.cam.lookat[1] = 0.7
            viewer.cam.lookat[2] = 0
            self.env.viewer = viewer

    def step(self, action):
        reward_sum = 0
        done = False
        if self.demonstration_phase:
            self.env._partially_observable = False
            for i in range(self.demonstration_action_repeat):
                self.observation, _, _, info = self.env.step(self.oracle_policy.get_action(self.observation))
                self.step_in_trial += 1
                self.demonstration_success = self.demonstration_success or info['success']

            if self.trial_timeout():
                self.demonstration_successes += self.demonstration_success
                self.demonstration_phase = False
                self._reset_env()
        else:
            self.env._partially_observable = self.partially_observable
            for i in range(self.action_repeat):
                self.observation, reward, done, info = self.env.step(action)
                self.step_in_trial += 1
                reward_sum += reward
                # if not self.trial_success and info['success']:
                    # print(f'trial {self.trial_in_episode} step {self.step_in_trial}success {self.trial_success} action repeat {self.action_repeat}')
                self.trial_success = self.trial_success or info['success']

            if self.trial_timeout():
                # print(f'trial {self.trial_in_episode} success {self.trial_success}')
                self.num_successful_trials += self.trial_success
                self.demonstration_phase = not self.trial_success and self.demonstrations
                self.trial_in_episode += 1
                self._reset_env()

        if self.trial_in_episode >= self.max_trials_per_episode:
            done = True
            self.trial_in_episode -= 1
        info = self.append_env_info(info)
        full_observation = self.get_full_observation(self.observation, action, reward_sum)
        return full_observation, reward_sum, done, info

    def render(self, *args, **kwargs):
        self.env.render(*args, **kwargs)

    def append_env_info(self, info):
        for env_name in self.benchmark.all_possible_classes.keys():
            info[env_name.replace('-', '') + '_episode_success'] = float('nan')

        info[self.env_name.replace('-', '') + '_episode_success'] = self.num_successful_trials / self.max_trials_per_episode
        info['episode_success'] = self.num_successful_trials / self.max_trials_per_episode
        if self.env_name in self.benchmark.TRAIN_CLASSES.keys():
            info['training_episode_success'] = self.num_successful_trials / self.max_trials_per_episode
            info['testing_episode_success'] = float('nan')
        elif self.env_name in self.benchmark.TEST_CLASSES.keys():
            info['training_episode_success'] = float('nan')
            info['testing_episode_success'] = self.num_successful_trials / self.max_trials_per_episode

        info['demonstration_success'] = self.demonstration_successes
        return info

    def trial_timeout(self):
        if self.v2:
            return self.step_in_trial >= self.env.max_path_length / 2
        else:
            # print(f'step {self.step_in_trial} max {self.env.max_path_length}')
            return self.step_in_trial >= self.env.max_path_length / 2

    def get_full_observation(self, env_obs, prev_action=None, prev_reward=None):
        if self.demonstration_phase or prev_action is None:
            prev_action = np.zeros_like(self.action_space.sample())

        prev_reward = prev_reward or 0
        # time = (self.step_in_trial + (self.trial_in_episode * self.env.max_path_length)) /  \
        #        (self.env.max_path_length * self.max_trials_per_episode)
        time = self.step_in_trial / self.env.max_path_length
        task_info = prev_reward if self.dense_rewards else (self.trial_success and not self.demonstration_phase)
        state = np.concatenate((env_obs, [self.demonstration_phase, time, task_info]))
        if self.prev_action_obs:
            state = np.concatenate((state, prev_action))
        if self.visual_observations:
            self.env.viewer.render(64, 64, None)
            camera_image, depth = self.env.viewer.read_pixels(64, 64, depth=True)
            # camera_image = np.concatenate((camera_image, depth.reshape(128, 128, 1)), axis=2)[::-1]
            # camera_image = depth[::-1]
            camera_image = np.transpose(camera_image[::-1], (2, 0, 1))
            camera_image = camera_image / 255.0 - 0.5
            return dict(state=state,
                    camera_image=camera_image)
            # return camera_image
        else:
            full_obs = dict(
                state=state,
                task_id=np.array(self.env_id),
                demonstration_phase=np.array(self.demonstration_phase),
            )
        return full_obs


if __name__ == '__main__':
    s = time.time()
    env = GeneralizedMetaWorld(benchmark='ml10', action_repeat=1, demonstration_action_repeat=1,
                               max_trials_per_episode=1, sample_num_classes=1, mode='meta_training', v2=True)
    # print('env init took: '+ str(time.time() - s))
    while True:
        s = time.time()
        obs = env.reset()  # Reset environment
        # print('reset took: ' + str(time.time() - s))
        done = False
        step = 0
        reward_sum = 0
        while not done:
            obs, reward, done, info = env.step(action)  # Step the environoment with the sampled random action
            reward_sum += reward
            # env.render(mode='human')
            step += 1
            # time.sleep(0.04)

        print('num steps: ' + str(step))
        # print('reward sum: ' + str(reward_sum))
        print(f'info {info}')
