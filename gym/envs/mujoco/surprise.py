import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class SurpriseEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'arm_claw.xml', 4)
        utils.EzPickle.__init__(self)

    def _step(self, a):
        self.do_simulation(a, self.frame_skip)
        
        ob = self._get_obs()
        reward = 0
        done = False
        return ob, reward, done, None

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 2.5
        self.viewer.cam.elevation += 20


    def reset_model(self):
        # raise NotImplementedError
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat,
            self.model.data.qvel.flat
        ])