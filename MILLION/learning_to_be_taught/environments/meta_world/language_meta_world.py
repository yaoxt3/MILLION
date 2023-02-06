from learning_to_be_taught.environments.meta_world.generalized_meta_world import GeneralizedMetaWorld
from torchtext.vocab import GloVe
import gym
import numpy as np
import pickle
import os
import random


class LanguageMetaWorld(GeneralizedMetaWorld):
    EMBED_DIM = 50

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.instruction_index = 0
        self.EMBED_DIM = 50
        vocab_path = os.path.join(os.path.dirname(__file__), 'meta_world_vocab.pkl')
        assert os.path.exists(vocab_path), 'please run save_used_word_embeddings.py before using LanguageMetaWorld'
        self.word_mapping = pickle.load(open(vocab_path, 'rb'))
        if self.visual_observations:
            self.observation_space = gym.spaces.Dict({
                'camera_image': gym.spaces.Box(low=-1, high=1, shape=(3, 64, 64)),
                'state': gym.spaces.Box(low=-10, high=10, shape=(self.EMBED_DIM,)),
            })

            # self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(3, 64, 64))
        else:
            self.observation_space = gym.spaces.Dict({
                'state': gym.spaces.Box(low=-10, high=10, shape=(self.EMBED_DIM,)),
                'task_id': gym.spaces.Discrete(50),
                'demonstration_phase': gym.spaces.Discrete(2),
            })

    # Task: use the opposite actions to deal with the similar
    # task. For example, if you learn how to open a drawer from people’s
    # demonstrations, you can learn how to close the drawer from the opening actions
    # Through the opposition-based learning, we can provide
    # MDN with more information about the meta-test tasks to improve the
    # performance of the meta test.
    # Here (language instructions): generate antonym verb
    # Case with task demonstrations: use oppositon-based RL

    # def reset(self, instruction=None):
    #     self._reset_env(new_episode=True, instruction=instruction)
    #     self.trial_in_episode = 0
    #     self.demonstration_phase = True
    #     self.num_successful_trials = 0
    #     self.demonstration_successes = 0
    #     return self.get_full_observation(self.observation)

    def _reset_env(self, *args, instruction=None, **kwargs):
        super()._reset_env(*args, **kwargs)
        self.instruction = random.sample(env_instructions[self.env_name], 1)[0]
        self.instruction = self.instruction.split()
        self.instruction_index = 0

    def step(self, action):
        reward_sum = 0
        if self.demonstration_phase:
            instruction, instruction_finished = self.get_instruction_word()
            full_observation = self.get_full_observation(instruction_word=instruction)
            self.demonstration_phase = not instruction_finished
            info = {}
        else:
            self.env._partially_observable = True
            for i in range(self.action_repeat):
                self.observation, reward, done, info = self.env.step(action)
                self.step_in_trial += 1
                reward_sum += reward
                self.trial_success = self.trial_success or info['success']

            if self.trial_timeout():
                self.num_successful_trials += self.trial_success
                self.demonstration_phase = not self.trial_success  # generate another demonstration if failed
                self.trial_in_episode += 1
                self._reset_env()
            full_observation = self.get_full_observation(self.observation)

        done = self.trial_in_episode >= self.max_trials_per_episode
        info = self.append_env_info(info)
        if done:
            self.valid_samples.append((full_observation, reward_sum, info))
            #self.opposite_samples.append(opp_observation)
        return full_observation, reward_sum, done, info

    def get_instruction_word(self):
        # print('old instruction')
        # np.set_printoptions(precision=2)
        # print(f'instruction is {self.instruction}')
        # if self.instruction_index == 0:
        #     self.instruction = random.sample(env_instructions[self.env_name], 1)[0].split()
        #     print(f'sampled new instruction {self.instruction}')
        word = self.instruction[self.instruction_index]
        if word == 'goal_pos':
            embedding = np.concatenate((self.env._get_pos_goal(), np.zeros(self.EMBED_DIM - 3)))
        else:
            embedding = self.word_mapping[word]
            # embedding = self.word_embedding.get_vecs_by_tokens(word).float()

        self.instruction_index = (self.instruction_index + 1) % len(self.instruction)
        instruction_finished = self.instruction_index == 0
        return embedding, instruction_finished

    def get_full_observation(self, env_obs=None, instruction_word=None):
        if env_obs is not None:
            state = np.concatenate((env_obs, [self.demonstration_phase, self.step_in_trial / 150,
                                              self.trial_success and not self.demonstration_phase]))
            random_concat_values = np.random.default_rng().uniform(-1,1,self.EMBED_DIM - state.shape[0])
            zero_values = np.zeros(self.EMBED_DIM - state.shape[0])
            # The concatenation of a randomly initialized vector could improve the model’s
            # performance, as the added zeros decrease the dimensions of the given
            # instructions.
            state = np.concatenate((state, random_concat_values))
        elif instruction_word is not None:
            state = np.array(instruction_word)

        if self.visual_observations:
            self.env.viewer.render(64, 64, None)
            camera_image, depth = self.env.viewer.read_pixels(64, 64, depth=True)
            # camera_image = np.concatenate((camera_image, depth.reshape(128, 128, 1)), axis=2)[::-1]
            # camera_image = depth[::-1]
            camera_image = np.transpose(camera_image[::-1], (2, 0, 1))
            camera_image = camera_image / 128 - 1
            return dict(state=state,
                    camera_image=camera_image)
            #return camera_image
        else:
            camera_image = None

        full_obs = dict(
            state=state,
            task_id=np.array(self.env_id),
            demonstration_phase=np.array(int(self.demonstration_phase)),
            # camera_image=camera_image
        )

        return full_obs


env_instructions = {
    "reach-v1": ["reach to goal_pos", 'reach goal_pos'],
    'push-v1': ["push goal_pos", 'push to goal_pos', 'push object to goal_pos', 'push block to goal_pos'],
    'pick-place-v1': ["pick and place at goal_pos", 'pick object and place at goal_pos'],
    'door-open-v1': ["pull goal_pos", 'open door', 'pull to goal_pos'],
    'drawer-open-v1': ["pull goal_pos", 'pull to goal_pos', 'pull back to goal_pos'],
    'drawer-close-v1': ["push goal_pos", 'push to goal_pos', 'push forward to goal_pos'],
    'button-press-topdown-v1': ["push object down to goal_pos", 'press button', 'press down', 'press button down'],
    'peg-insert-side-v1': [""],
    'window-open-v1': ['push right to goal_pos', 'push object right', 'slide object left', 'open window'],
    'window-close-v1': ['push left to goal_pos', 'push object left', 'slide object right', 'close window'],
    'door-close-v1': ['push to goal_pos', 'close door', 'push from left'],
    'reach-wall-v1': ['reach over obstacle to gonlp model to detect verb and get its antonymal_pos', 'reach goal_pos', 'reach to goal_pos'],
    'pick-place-wall-v1': ['pick object and place at goal_pos', 'pick and place at goal_pos'],
    'push-wall-v1': ['push object around obstacle to goal_pos', 'push to goal_pos', 'push object to goal_pos'],
    'button-press-v1': ['press forward', 'press button forward', 'push to goal_pos'],
    'button-press-topdown-wall-v1': ['press behind obstacle', 'press down to goal_pos', 'press down'],
    'button-press-wall-v1': ['press button behind obstacle', 'press forward', 'press forward to goal_pos'],
    'peg-unplug-side-v1': ['pull object to right', 'pull object to goal_pos', 'pick object and pull to right'],
    'disassemble-v1': ['pick and pull up', 'pick and place at goal_pos', 'pick and put down at goal_pos'],
    'hammer-v1': ['push to goal_pos with object', 'push to goal_pos with hammer', 'use object to push to goal_pos'],
    'plate-slide-v1': ['push to goal_pos', 'push plate to goal_pos', 'push forward to goal_pos'],
    'plate-slide-side-v1': ['push left to goal_pos', 'push plate left to goal_pos', 'slide left to goal_pos'],
    'plate-slide-back-v1': ['push back to goal_pos', 'push plate back to goal_pos', 'slide back to goal_pos'],
    'plate-slide-back-side-v1': ['push right to goal_pos', 'push plate right to goal_pos', 'slide right to goal_pos'],
    'handle-press-v1': ['push down', 'press down', 'push down to goal_pos', 'press down to goal_pos'],
    'handle-pull-v1': ['pull up', 'push up', 'push up to goal_pos', 'pull up to goal_pos'],
    'handle-press-side-v1': ['push down', 'press down', 'push down to goal_pos', 'press down to goal_pos'],
    'handle-pull-side-v1': ['pull up', 'push up', 'push up to goal_pos', 'pull up to goal_pos'],
    'stick-push-v1': ['push with object to goal_pos', 'use stick to push to goal_pos', 'push with stick to goal_pos'],
    'stick-pull-v1': ['push with object to goal_pos', 'use stick to push to goal_pos', 'push with stick to goal_pos'],
    'basketball-v1': ['pick and place at goal_pos', 'pick ball and place at goal_pos', 'pick ball and place at goal_pos from the top'],
    'soccer-v1': ['push to goal_pos', 'push ball to goal_pos', 'place object at goal_pos', 'place ball at goal_pos'],
    'faucet-open-v1': ['open faucet', 'push to goal_pos', 'push right', 'push from left to right',
                       'turn object from left to right', 'turn to goal_pos'],
    'faucet-close-v1': ['close faucet', 'push to goal_pos', 'push left', 'push from right to left',
                        'turn object from right to left', 'turn to goal_pos'],
    'coffee-push-v1': ['push object to goal_pos', 'push cup to goal_pos', 'push object forward to goal_pos'],
    'coffee-pull-v1': ['place object away', 'pick and place at goal_pos', 'place cup away'],
    'coffee-button-v1': ['push forward', 'press button', 'press coffee button'],
    'sweep-v1': ['sweep', 'sweep object from table', 'sweep block from table', 'slide object from table', 'slide block from table'],
    'sweep-into-v1': ['sweep into hole', 'push to goal_pos', 'push block to goal_pos', 'push block into hole'],
    'pick-out-of-hole-v1': ['pick object', 'pick object out of hole', 'pick and place at goal_pos'],
    'assembly-v1': ['pick and place at goal_pos', 'assemble', 'pick object and place down at goal_pos'],
    'shelf-place-v1': ['pick and place at goal_pos', 'place on shelf at goal_pos', 'pick object and place on shelf at goal_pos'],
    'push-back-v1': ['push back to goal_pos', 'push to goal_pos', 'puch block to goal_pos',
                     'push block back to goal_pos'],
    'lever-pull-v1': ['pull to goal_pos'],
    'dial-turn-v1': ['turn object', 'dial turn', 'rotate object'],

    'bin-picking-v1': ['pick object', 'pick and place at goal_pos'],
    'box-close-v1': ['pick object and place at goal_pos', 'pick and place at goal_pos'],
    'hand-insert-v1': ['sweep object to goal_pos', 'pick object and place at goal_pos', 'push object to goal_pos',
                       'push block to goal_pos'],
    'door-lock-v1': ['push object down to goal_pos', 'turn object down to goal_pos'],
    'door-unlock-v1': ['push object to goal_pos', 'turn object to goal_pos'],
}

if __name__ == '__main__':
    # env = LanguageMetaWorld(benchmark='reach-v1', action_repeat=1, demonstration_action_repeat=1,
                            # max_trials_per_episode=1, sample_num_classes=1, visual_observations=False)
    env = LanguageMetaWorld(benchmark='ml45', action_repeat=1, demonstration_action_repeat=5,
                            max_trials_per_episode=3, sample_num_classes=5, mode='meta_testing')
    
    while True:
        obs = env.reset()  # Reset environment
        done = False
        step = 0
        reward_sum = 0
        while not done:
            print(f'obs {obs}')
            action = env.action_space.sample()  # Sample an action
            obs, reward, done, info = env.step(action)  # Step the environoment with the sampled random action
            # outfile = open('img.pkl','wb')
            # pickle.dump(obs['camera_image'], outfile)
            reward_sum += reward
            # env.render(mode='human')
            step += 1
            # time.sleep(0.04)

        print('num steps: ' + str(step))
        # print('reward sum: ' + str(reward_sum))
        print(f'info {info}')
