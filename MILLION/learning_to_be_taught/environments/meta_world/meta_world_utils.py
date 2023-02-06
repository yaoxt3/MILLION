
from metaworld.envs.mujoco.sawyer_xyz import *
import numpy as np
import random
from metaworld import Benchmark, _ML_OVERRIDE, _encode_task
from collections import OrderedDict


def _make_tasks(classes, args_kwargs, kwargs_override, num_goals=5):
    import multiprocessing as mp

    def square(env_name, args, env_queue, task_queue):
        assert len(args['args']) == 0
        env_cls = classes[env_name]
        env = env_cls()
        env._freeze_rand_vec = False
        env._set_task_called = True
        rand_vecs = []
        kwargs = args['kwargs'].copy()
        del kwargs['task_id']
        env._set_task_inner(**kwargs)
        for _ in range(num_goals):
            env.reset()
            rand_vecs.append(env._last_rand_vec)
        unique_task_rand_vecs = np.unique(np.array(rand_vecs), axis=0)
        assert unique_task_rand_vecs.shape[0] == num_goals

        env.close()
        for rand_vec in rand_vecs:
            kwargs = args['kwargs'].copy()
            del kwargs['task_id']
            kwargs.update(dict(rand_vec=rand_vec, env_cls=env_cls))
            kwargs.update(kwargs_override)
            # tasks.append(_encode_task(env_name, kwargs))
            task_queue.put(_encode_task(env_name, kwargs))
        env_queue.put((env_name, env))

    env_queue = mp.Queue(maxsize=len(classes))
    task_queue = mp.Queue(maxsize=num_goals * len(classes))
    processes = [mp.Process(target=square, args=(env_name, args, env_queue, task_queue))
                 for (env_name, args) in args_kwargs.items()]
    for p in processes:
        p.start()

    tasks = []
    envs = []
    for i in range(len(classes) * num_goals):
        tasks.append(task_queue.get())

    for i in range(len(classes)):
        envs.append(env_queue.get())

    for p in processes:
        p.join()

    return envs, tasks
