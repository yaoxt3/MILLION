from metaworld import Benchmark, _env_dict, _make_tasks, _ML_OVERRIDE
import metaworld
import random
import copy
from abc import abstractmethod
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_peg_insertion_side_v2 import SawyerPegInsertionSideEnvV2
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_lever_pull_v2 import SawyerLeverPullEnvV2
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_handle_press_side_v2 import SawyerHandlePressSideEnvV2

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
# HARD_MODE_CLS_DICT['test'].pop('hand-insert-v1') #reward function not working


class DemonstrationBenchmark(Benchmark):
    def __init__(self, mode='meta_training', sample_num_classes=1):
        assert mode in ['all', 'meta_training', 'meta_testing']
        super().__init__()
        if mode == 'meta_training':
            classes = copy.deepcopy(self.TRAIN_CLASSES)
            kwargs = self.train_kwargs
        elif mode == 'meta_testing':
            classes = copy.deepcopy(self.TEST_CLASSES)
            kwargs = self.test_kwargs
        elif mode == 'all':
            classes = copy.deepcopy(self.TRAIN_CLASSES)
            classes.update(self.TEST_CLASSES)
            kwargs = self.train_kwargs
            kwargs.update(self.test_kwargs)
        sample_num_classes = min(len(classes.keys()), sample_num_classes)
        self.classes_keys = random.sample(list(classes.keys()), sample_num_classes)
        self.classes = {key: classes[key] for key in self.classes_keys}
        self.all_possible_classes = copy.deepcopy(self.TRAIN_CLASSES)
        self.all_possible_classes.update(copy.deepcopy(self.TEST_CLASSES))
        self.class_to_id = {name: i for name, i in zip(classes, range(len(self.all_possible_classes)))}
        self.kwargs = dict((key, value) for key, value in kwargs.items() if key in self.classes)
        self._tasks = _make_tasks(self.classes,
                                 self.kwargs,
                                 _ML_OVERRIDE)
        self._envs = [(env_name, self.classes[env_name]()) for env_name in self.classes_keys]

    def sample_env_and_task(self):
        env_id = random.choice(range(len(self._envs)))
        env_name, env = self._envs[env_id]
        global_env_id = self.class_to_id[env_name]

        task = random.choice([task for task in self._tasks if task.env_name == env_name])
        return env_name, env, task, global_env_id

    @property
    @abstractmethod
    def TRAIN_CLASSES(self) -> 'OrderedDict[EnvName, Type]':
        """Get all of the environment classes used for training."""
        pass

    @property
    @abstractmethod
    def TEST_CLASSES(self) -> 'OrderedDict[EnvName, Type]':
        """Get all of the environment classes used for testing."""
        pass

    @property
    @abstractmethod
    def train_kwargs(self):
        pass

    @property
    @abstractmethod
    def test_kwargs(self):
        pass

class SingleEnv(DemonstrationBenchmark):
    # train_kwargs = _env_dict.medium_mode_train_args_kwargs
    # test_kwargs = _env_dict.medium_mode_test_args_kwargs
    train_kwargs = _env_dict.HARD_MODE_ARGS_KWARGS['train']
    test_kwargs = _env_dict.HARD_MODE_ARGS_KWARGS['test']

    def __init__(self, env_name, *args, **kwargs):
        self.env_name = env_name
        super().__init__(*args, **kwargs)

    @property
    def TRAIN_CLASSES(self) -> 'OrderedDict[EnvName, Type]':
        classes = dict((name, env) for name, env in _env_dict.HARD_MODE_CLS_DICT['train'].items() if
                       name == self.env_name)
        return classes

    @property
    def TEST_CLASSES(self) -> 'OrderedDict[EnvName, Type]':
        # mode = 'train' if self.meta_training else 'test'
        classes = dict((name, env) for name, env in _env_dict.HARD_MODE_CLS_DICT['test'].items() if
                       name == self.env_name)
        return classes

class ML1(DemonstrationBenchmark):
    train_kwargs = _env_dict.medium_mode_train_args_kwargs
    test_kwargs = _env_dict.medium_mode_test_args_kwargs

    @property
    def TRAIN_CLASSES(self) -> 'OrderedDict[EnvName, Type]':
        classes = dict((name, env) for name, env in _env_dict.EASY_MODE_CLS_DICT.items() if
                       name in ['reach-v1', 'push-v1', 'pick-place-v1'])
        return classes

    @property
    def TEST_CLASSES(self) -> 'OrderedDict[EnvName, Type]':
        return self.TRAIN_CLASSES

class ML10(DemonstrationBenchmark):
    TRAIN_CLASSES = MEDIUM_MODE_CLS_DICT['train']
    TEST_CLASSES = MEDIUM_MODE_CLS_DICT['test']
    train_kwargs = _env_dict.medium_mode_train_args_kwargs
    test_kwargs = _env_dict.medium_mode_test_args_kwargs

class ML45(DemonstrationBenchmark):
    TRAIN_CLASSES = HARD_MODE_CLS_DICT['train']
    TEST_CLASSES = HARD_MODE_CLS_DICT['test']
    train_kwargs = _env_dict.HARD_MODE_ARGS_KWARGS['train']
    test_kwargs = _env_dict.HARD_MODE_ARGS_KWARGS['test']
