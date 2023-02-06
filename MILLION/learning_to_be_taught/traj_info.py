from rlpyt.samplers.collections import TrajInfo


class EnvInfoTrajInfo(TrajInfo):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def step(self, observation, action, reward, done, agent_info, env_info):
        super().step(observation, action, reward, done, agent_info, env_info)

        # log all int, float and bools from env_info
        for key, value in env_info._asdict().items():
            try:
                float_value = float(value)
                self.__setattr__(key, float_value)
                if not hasattr(self, key + '_traj_sum'):
                    setattr(self, key + '_traj_sum', float_value)
                else:
                    new_value = self.__getattribute__(key + '_traj_sum') + float_value
                    setattr(self, key + '_traj_sum', new_value)
            except (ValueError, TypeError):
                # can't log value
                pass
