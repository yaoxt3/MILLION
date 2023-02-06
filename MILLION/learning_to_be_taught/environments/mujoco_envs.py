import gym

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env, ReacherEnv


class ImitationReacherEnv(ReacherEnv):
    def __init__(self):
        # utils.EzPickle.__init__(self)
        ReacherEnv.__init__(self)

    def _step(self, a):
        vec = self.get_body_com("fingertip")-self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + reward_ctrl
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    # def reset_model(self):
    #     qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
    #     while True:
    #         self.goal = self.np_random.uniform(low=-.2, high=.2, size=2)
    #         if np.linalg.norm(self.goal) < 2:
    #             break
    #     qpos[-2:] = self.goal
    #     qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
    #     qvel[-2:] = 0
    #     self.set_state(qpos, qvel)
    #     return self._get_obs()
    #
    def _get_obs(self):
        theta = self.data.qpos.flat[:2]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.data.qpos.flat[2:],
            self.data.qvel.flat[:2],
            self.get_body_com("fingertip") - self.get_body_com("target")
        ])




if __name__ == '__main__':
    env = ImitationReacherEnv()
    # env = gym.make('Reacher-v2')
    while True:
        done = False
        obs = env.reset()
        print('reset ###################')
        while not done:
            # action = env.oracle(obs)
            action = env.action_space.sample()
            action = [0, 0]
            obs, reward, done, info = env.step(action)
            print('reward: ' + str(reward))
            env.render()