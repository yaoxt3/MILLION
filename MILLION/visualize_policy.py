import argparse
import gym
import torch
import time
import numpy as np
from rlpyt.envs.gym import GymEnvWrapper, EnvInfoWrapper
from rlpyt.utils.buffer import torchify_buffer, buffer_from_example, numpify_buffer
from rlpyt.agents.pg.categorical import CategoricalPgAgent
from learning_to_be_taught.vmpo.models import FfModel, CategoricalFfModel, TransformerModel, FfSharedModel
from rlpyt.agents.pg.mujoco import  MujocoLstmAgent, MujocoFfAgent
from learning_to_be_taught.environments.meta_world.meta_world import MetaWorld
from learning_to_be_taught.environments.meta_world.generalized_meta_world import GeneralizedMetaWorld
from learning_to_be_taught.environments.meta_world.language_meta_world import LanguageMetaWorld
from rlpyt.agents.qpg.sac_agent import SacAgent
from learning_to_be_taught.recurrent_sac.efficient_recurrent_sac_agent import EfficientRecurrentSacAgent
from learning_to_be_taught.environments.pendulum import Pendulum
from learning_to_be_taught.recurrent_sac.recurrent_sac_agent import RecurrentSacAgent
from learning_to_be_taught.recurrent_sac.transformer_model import PiTransformerModel, QTransformerModel
from learning_to_be_taught.behavioral_cloning.behavioral_cloning_agent import BehavioralCloningAgent
from rlpyt.agents.pg.mujoco import MujocoLstmAgent, MujocoFfAgent
from learning_to_be_taught.vmpo.gaussian_vmpo_agent import MujocoVmpoAgent
from learning_to_be_taught.vmpo.models import TransformerModel, GeneralizedTransformerModel
from learning_to_be_taught.vmpo.compressive_transformer import CompressiveTransformer
from learning_to_be_taught.vmpo.models import FfModel, CategoricalFfModel


def simulate_policy(env, agent, render):
    obs = env.reset()
    print("AAA: ", obs)
    agent.to_device(0)
    observation = buffer_from_example(obs, 1)
    loop_time = 0.01
    returns = []
    mses = []
    successes = []
    while True:
        observation[0] = env.reset()
        action = buffer_from_example(env.action_space.null_value(), 1)
        reward = np.zeros(1, dtype="float32")
        obs_pyt, act_pyt, rew_pyt = torchify_buffer((observation, action, reward))
        agent.reset()
        print(f'env name {env.env_name}')
        done = False
        step = 0
        reward_sum = 0
        # env.render()
        # time.sleep(1.1)
        forward_reward = 0
        while not done:
            loop_start = time.time()
            step += 1
            # print(f'obs {obs_pyt}')
            act_pyt, agent_info = agent.step(obs_pyt, act_pyt, rew_pyt)
            action = numpify_buffer(act_pyt)[0]
            # print('action: '+ str(action))
            # action = np.argmax(observation[0].demonstration_actions)
            # print(np.argmax(obs_pyt[0].demonstration_actions) == action)
            # print(f'action : {action} step {step}')
            obs, reward, done, info = env.step(action)
            reward_sum += reward
            # print('reward sum: ' + str(reward_sum))
            observation[0] = obs
            rew_pyt[0] = float(reward)
            sleep_time = loop_time - (time.time() - loop_start)
            sleep_time = 0 if (sleep_time < 0) else sleep_time
            if render:
                time.sleep(sleep_time)
                env.render(mode='human')

        # if info.demonstration_success > 0:
        successes.append(info.episode_success)
        print('episode success: ' + str(info.episode_success) + 'avg success: ' + str(sum(successes)/ len(successes)))
        returns.append(reward_sum)
        print('avg return: ' + str(sum(returns) / len(returns)) + ' return: ' + str(reward_sum) + '  num_steps: ' + str(
            step))
        # print(f'forward reward: {forward_reward}')


def make_env(**kwargs):
    info_example = {'timeout': 0}
    # env = MetaWorld(benchmark='ml10', demonstrations_flag=True, action_repeat=1)
    # env = GeneralizedMetaWorld(benchmark='ml10', action_repeat=2, demonstration_action_repeat=5, sample_num_classes=1,
                               # mode='meta_training', max_trials_per_episode=3, dense_rewards=True, demonstrations=False)
    env = LanguageMetaWorld(benchmark='reach-v1', action_repeat=2, mode='all', max_trials_per_episode=3, sample_num_classes=5)
    # env = EasyReacher(demonstrations_flag=True)
    # env = Pendulum(demonstrations_flag=False)
    # env = gym.make(**kwargs)
    env = GymEnvWrapper(EnvInfoWrapper(env, info_example))
    # env = Monitor(env, './logs/Videos', force=True, write_upon_reset=True)
    return env


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path', help='path to params.pkl',
                        default='/home/Trained_Policy/ml10/language_instructions/run_3/params.pkl')
                        # default='/home/alex/learning_to_be_taught/logs/ml45_language_small_obs_new/params.pkl')
    parser.add_argument('--env', default='HumanoidPrimitivePretraining-v0',
                        choices=['HumanoidPrimitivePretraining-v0', 'TrackEnv-v0'])
    parser.add_argument('--algo', default='ppo', choices=['sac', 'ppo'])
    args = parser.parse_args()

    snapshot = torch.load(args.path, map_location=torch.device('cpu'))
    agent_state_dict = snapshot['agent_state_dict']
    env = make_env(id='Ant-v3')
    # env = GridEnv(render=True)

    # agent = DqnAgent(ModelCls=DemonstrationQModel, model_kwargs=dict(input_size=16), eps_eval=0)
    # agent = MetaImitationAgent(ModelCls=DemonstrationAttentionDotProductQModel,
    #                            initial_model_state_dict=agent_state_dict)
    # agent = BehavioralCloningAgent(model_kwargs=dict(n_head=4, d_model=128, dim_feedforward=256, num_encoder_layers=4, num_decoder_layers=4))
    # agent = RecurrentSacAgent(ModelCls=FakeRecurrentPiModel, QModelCls=FakeRecurrentQModel)
    # agent = RecurrentSacAgent(ModelCls=PiTransformerModel, QModelCls=QTransformerModel,
    #                           q_model_kwargs=dict(size='small', state_action_input=True), model_kwargs=dict(size='small'))
    # agent = EfficientRecurrentSacAgent()
    # agent = MujocoFfAgent(ModelCls=FfModel)
    # agent = CategoricalPgAgent(ModelCls=CategoricalFfModel, model_kwargs=dict(observation_shape=(6,), action_size=3))

    # agent = SacAgent()
    # agent = MujocoLstmAgent(ModelCls=TransformerModel, model_kwargs=dict(size='small'))
    # agent = MujocoLstmAgent(ModelCls=FfSharedModel, model_kwargs=dict(linear_value_output=False, full_covariance=False))
    # agent = MujocoVmpoAgent(ModelCls=FfSharedModel, model_kwargs=dict(linear_value_output=False, full_covariance=False))
    # agent = MujocoVmpoAgent(ModelCls=GeneralizedTransformerModel, model_kwargs=dict(linear_value_output=False,
    #                                                                                 size='small',
    #                                                                                 episode_length=150,
    #                                                                                 demonstration_length=50,
    #                                                                                 layer_norm=False,
    #                                                                                 seperate_value_network=False))

    #ml10 benchmark suquence_length=64(language)
    agent = MujocoVmpoAgent(ModelCls=CompressiveTransformer, model_kwargs=dict(linear_value_output=False,
                                                                               size='medium', sequence_length=64,
                                                                               seperate_value_network=False,
                                                                               observation_normalization=False))
    #print(f'{env.spaces}')
    agent.initialize(env_spaces=env.spaces)
    agent.load_state_dict(agent_state_dict)
    agent.eval_mode(1)
    # agent.sample_mode(0)
    from rlpyt.samplers.serial.collectors import SerialEvalCollector
    from rlpyt.samplers.collections import TrajInfo

    eval_collector = SerialEvalCollector(envs=[env],
                                         agent=agent,
                                         TrajInfoCls=TrajInfo,
                                         max_T=10000)
    # x  = eval_collector.collect_evaluation(0)
    simulate_policy(env, agent, render=True)
