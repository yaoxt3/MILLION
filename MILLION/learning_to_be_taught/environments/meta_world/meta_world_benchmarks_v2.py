from metaworld import Benchmark, _env_dict, _make_tasks, _ML_OVERRIDE
# from metaworld import ML10_V2 as ml10_v2_class_dict
import metaworld
import random
import copy
from abc import abstractmethod
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_peg_insertion_side_v2 import SawyerPegInsertionSideEnvV2
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_lever_pull_v2 import SawyerLeverPullEnvV2
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_handle_press_side_v2 import SawyerHandlePressSideEnvV2
from learning_to_be_taught.environments.meta_world.meta_world_benchmarks import DemonstrationBenchmark

MEDIUM_MODE_CLS_DICT = copy.deepcopy(_env_dict.MEDIUM_MODE_CLS_DICT)
MEDIUM_MODE_CLS_DICT['train'].pop('sweep-v1') #reward function not working
MEDIUM_MODE_CLS_DICT['train'].pop('peg-insert-side-v1') #demonstration not working
# MEDIUM_MODE_CLS_DICT['test'].pop('lever-pull-v1') #demonstration not working

HARD_MODE_CLS_DICT = copy.deepcopy(_env_dict.HARD_MODE_CLS_DICT)
# HARD_MODE_CLS_DICT['train']['handle-press-side-v1'] = SawyerHandlePressSideEnvV2
HARD_MODE_CLS_DICT['train'].pop('sweep-v1') #reward function not working
HARD_MODE_CLS_DICT['train'].pop('peg-insert-side-v1') #demonstration policy not working
HARD_MODE_CLS_DICT['train'].pop('lever-pull-v1') #reward function not working
HARD_MODE_CLS_DICT['test'].pop('bin-picking-v1') #reward function not working


class SingleEnvV2(DemonstrationBenchmark):
    # train_kwargs = _env_dict.medium_mode_train_args_kwargs
    # test_kwargs = _env_dict.medium_mode_test_args_kwargs
    # train_kwargs = _env_dict.HARD_MODE_ARGS_KWARGS['train']
    # test_kwargs = _env_dict.HARD_MODE_ARGS_KWARGS['test']
    train_kwargs = _env_dict.ml10_train_args_kwargs # medium_mode_train_args_kwargs
    test_kwargs = _env_dict.ml10_test_args_kwargs # medium_mode_test_args_kwargs

    def __init__(self, env_name, *args, **kwargs):
        self.env_name = env_name
        super().__init__(*args, **kwargs)

    @property
    def TRAIN_CLASSES(self) -> 'OrderedDict[EnvName, Type]':
        classes = dict((name, env) for name, env in _env_dict.ALL_V2_ENVIRONMENTS.items() if
                       name == self.env_name)
        return classes

    @property
    def TEST_CLASSES(self) -> 'OrderedDict[EnvName, Type]':
        # mode = 'train' if self.meta_training else 'test'
        classes = dict((name, env) for name, env in _env_dict.ALL_V2_ENVIRONMENTS.items() if
                       name == self.env_name)
        return classes

class ML1V2(DemonstrationBenchmark):
    train_kwargs = _env_dict.ml10_train_args_kwargs # medium_mode_train_args_kwargs
    test_kwargs = _env_dict.ml10_test_args_kwargs # medium_mode_test_args_kwargs

    @property
    def TRAIN_CLASSES(self) -> 'OrderedDict[EnvName, Type]':
        classes = dict((name, env) for name, env in _env_dict.ML10_V2['train'].items() if
                       name in ['reach-v2', 'push-v2', 'pick-place-v2'])
        return classes

    @property
    def TEST_CLASSES(self) -> 'OrderedDict[EnvName, Type]':
        return self.TRAIN_CLASSES

class ML10V2(DemonstrationBenchmark):
    TRAIN_CLASSES = _env_dict.ML10_V2['train']
    TEST_CLASSES = _env_dict.ML10_V2['test']
    train_kwargs = _env_dict.ml10_train_args_kwargs # medium_mode_train_args_kwargs
    test_kwargs = _env_dict.ml10_test_args_kwargs # medium_mode_test_args_kwargs

class ML45V2(DemonstrationBenchmark):
    TRAIN_CLASSES = HARD_MODE_CLS_DICT['train']
    TEST_CLASSES = HARD_MODE_CLS_DICT['test']
    train_kwargs = _env_dict.HARD_MODE_ARGS_KWARGS['train']
    test_kwargs = _env_dict.HARD_MODE_ARGS_KWARGS['test']
