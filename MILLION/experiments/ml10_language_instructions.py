import argparse
import GPUtil
import multiprocessing
import torch
from typing import Dict
from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.samplers.async_.cpu_sampler import AsyncCpuSampler
from rlpyt.runners.async_rl import AsyncRlEval
from rlpyt.utils.launching.affinity import affinity_from_code
from rlpyt.utils.launching.variant import load_variant, update_config
from rlpyt.utils.launching.affinity import make_affinity
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.envs.gym import GymEnvWrapper
from rlpyt.runners.minibatch_rl import MinibatchRlEval
from traj_info import EnvInfoTrajInfo
from logger_context import config_logger
from learning_to_be_taught.environments.meta_world.generalized_meta_world import GeneralizedMetaWorld
from learning_to_be_taught.environments.meta_world.language_meta_world import LanguageMetaWorld
from learning_to_be_taught.environments.env_info_wrapper import EnvInfoWrapper
from learning_to_be_taught.vmpo.v_mpo import VMPO
from learning_to_be_taught.vmpo.multitask_vmpo import MultitaskAsyncVMPO, MultitaskVMPO
from rlpyt.samplers.async_.alternating_sampler import AsyncAlternatingSampler
from learning_to_be_taught.vmpo.gaussian_vmpo_agent import MujocoVmpoAgent, AlternatingVmpoAgent
from learning_to_be_taught.vmpo.compressive_transformer import CompressiveTransformer



def build_and_train(slot_affinity_code=None, log_dir='./data', run_ID=0,
                    serial_mode=False,
                    alternating_sampler=False,
                    snapshot: Dict = None,
                    config_update: Dict = None,
                    name='run'):
    # default configuration
    sequence_length = 64
    config = dict(
        vmpo_kwargs=dict(epochs=2, minibatches=1, discrete_actions=False, pop_art_reward_normalization=True,
                         T_target_steps=100),
        async_vmpo_kwargs=dict(epochs=4, discrete_actions=False, T_target_steps=100, batch_B=128, epsilon_eta=0.1,
                               batch_T=sequence_length, gae_lambda=1, num_tasks=50),
        sampler_kwargs=dict(batch_T=sequence_length, batch_B=22 * 12,
                            env_kwargs=dict(benchmark='ml10', action_repeat=2, demonstration_action_repeat=5,
                                            mode='meta_training', max_trials_per_episode=3),
                            eval_env_kwargs=dict(benchmark='ml10', action_repeat=2, demonstration_action_repeat=5,
                                                 mode='all', max_trials_per_episode=3),
                            eval_n_envs=22 * 4,
                            eval_max_steps=1e5,
                            eval_max_trajectories=22 * 4 * 4,
                            TrajInfoCls=EnvInfoTrajInfo),
        agent_kwargs=dict(ModelCls=CompressiveTransformer, model_kwargs=dict(sequence_length=sequence_length,
                                                                             size='medium',
                                                                             linear_value_output=False,
                                                                             seperate_value_network=False)),
        # agent_kwargs=dict(ModelCls=LstmModel, model_kwargs=dict(linear_value_output=False)),
        runner_kwargs=dict(n_steps=4e8, log_interval_steps=5e6),
        snapshot=snapshot,
        # algo='vmpo',
        algo='async_vmpo'
    )

    # update default config if available in log_dir or was provided as parameter
    try:
        variant = load_variant(log_dir)
        config = update_config(config, variant)
    except FileNotFoundError:
        if config_update is not None:
            config = update_config(config, config_update)

    if config['algo'] == 'vmpo':
        AlgoClass = VMPO
        SamplerClass = CpuSampler
        # SamplerClass = SerialSampler
        RunnerClass = MinibatchRlEval
        algo_kwargs = config['vmpo_kwargs']
        _async = False
    elif config['algo'] == 'async_vmpo':
        AlgoClass = MultitaskAsyncVMPO
        if alternating_sampler:
            SamplerClass = AsyncAlternatingSampler
        else:
            SamplerClass = AsyncCpuSampler
        RunnerClass = AsyncRlEval
        algo_kwargs = config['async_vmpo_kwargs']
        _async = True

    if serial_mode:
        SamplerClass = SerialSampler
        alternating_sampler = False
        config['sampler_kwargs']['batch_B'] = 4
        config['sampler_kwargs']['eval_n_envs'] = 4
        config['sampler_kwargs']['eval_max_trajectories'] = 4
        _async = False

    affinity = choose_affinity(slot_affinity_code, serial_mode, alternating_sampler, _async,
                               config['sampler_kwargs']['batch_B'])
    # continue training from saved state_dict if provided
    agent_state_dict = optimizer_state_dict = None
    if config['snapshot'] is not None:
        agent_state_dict = config['snapshot']['agent_state_dict']
        optimizer_state_dict = config['snapshot']['optimizer_state_dict']

    if alternating_sampler:
        AgentClass = AlternatingVmpoAgent
    else:
        AgentClass = MujocoVmpoAgent

    sampler = SamplerClass(
        **config['sampler_kwargs'],
        EnvCls=make_metaworld_env,
    )
    algo = AlgoClass(**algo_kwargs, initial_optim_state_dict=optimizer_state_dict)
    agent = AgentClass(initial_model_state_dict=agent_state_dict, **config['agent_kwargs'])
    runner = RunnerClass(
        **config['runner_kwargs'],
        algo=algo,
        agent=agent,
        sampler=sampler,
        affinity=affinity
    )
    config_logger(log_dir, name=name, snapshot_mode='last', log_params=config)
    runner.train()


def make_metaworld_env(**kwargs):
    info_example = {'timeout': 0}
    env = LanguageMetaWorld(**kwargs)
    return GymEnvWrapper(EnvInfoWrapper(env, info_example))


def choose_affinity(slot_affinity_code, serial_mode, alternating_sampler, async_mode, sampler_batch_B):
    if slot_affinity_code is None:
        num_cpus = multiprocessing.cpu_count()
        num_gpus = len(GPUtil.getGPUs())
        if serial_mode:
            affinity = make_affinity(n_cpu_core=num_cpus // 2, n_gpu=0, set_affinity=False)
        elif alternating_sampler:
            affinity = make_affinity(n_cpu_core=num_cpus, n_gpu=num_gpus, async_sample=True,
                                     optim_sample_share_gpu=True,
                                     alternating=True, set_affinity=True, hyperthread_offset=None)
            affinity['optimizer'][0]['cuda_idx'] = 1
            affinity['sampler'][0]['cuda_idx'] = 0
        elif async_mode:
            affinity = make_affinity(n_cpu_core=num_cpus, n_gpu=num_gpus, cpu_per_run=num_cpus,
                                     async_sample=True, optim_sample_share_gpu=False, set_affinity=True,
                                     alternating=False, n_socket=1)
            affinity['optimizer'][0]['cuda_idx'] = 0
        else:
            affinity = make_affinity(n_cpu_core=num_cpus, cpu_per_run=num_cpus, n_gpu=num_gpus, async_sample=False,
                                     set_affinity=False)
    else:
        affinity = affinity_from_code(slot_affinity_code)
    print(f'affinity: {affinity}')
    return affinity


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
    parser.add_argument('slot_affinity_code', nargs='?', default=None,
                        help='using all possible resources when not specified')
    parser.add_argument('log_dir_positional', nargs='?', help='required for automatic launching')
    parser.add_argument('run_id', nargs='?', help='required for automatic launching')
    parser.add_argument('--serial_mode', dest='serial_mode', action='store_true',
                        help='flag to run in serial mode is easier for debugging')
    parser.add_argument('--no_serial_mode', dest='serial_mode', action='store_false',
                        help='flag to run in serial mode is easier for debugging')
    parser.add_argument('--log_dir', required=False,
                        help='path to directory where log folder will be; Overwrites log_dir_positional')
    parser.add_argument('--snapshot_file', help='path to snapshot params.pkl containing state_dicts',
                        default=None)
    parser.add_argument('--name', help='path to snapshot params.pkl containing state_dicts',
                        default='run')

    args = parser.parse_args()
    log_dir = args.log_dir or args.log_dir_positional or './logs'
    print("training started with parameters: " + str(args))
    snapshot = None
    if args.snapshot_file is not None:
        snapshot = torch.load(args.snapshot_file, map_location=torch.device('cpu'))

    config_update = dict()

    build_and_train(slot_affinity_code=args.slot_affinity_code,
                    log_dir=log_dir,
                    run_ID=args.run_id,
                    snapshot=snapshot,
                    config_update=config_update,
                    name=args.name)
