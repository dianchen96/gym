import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class Box3dReachEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'arm_claw.xml', 4)
        utils.EzPickle.__init__(self)
        print self.init_qpos


    def _step(self, a):
        self.do_simulation(a, self.frame_skip)
        
        obs = self._get_obs()
        done = False
        
        contact_reward = 0.0

        # Check for contact
        d = self.unwrapped.data
        for coni in range(d.ncon):
            con = d.obj.contact[coni]
            if con.geom1 != 0 and con.geom2 == 12:
                # Small box is touched but not by table
                contact_reward = 1.0
        
        reward = contact_reward

        return obs, reward, done, dict(reward_contact=contact_reward)

    def viewer_setup(self):
        # self.viewer.cam.trackbodyid = 0
        self.viewer.cam.distance = self.model.stat.extent * 2.5
        self.viewer.cam.elevation += 20


    def reset_model(self):
        c = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv,)
        )
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat,
            self.model.data.qvel.flat
        ])


class Box3dGraspEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'arm_claw.xml', 4)
        utils.EzPickle.__init__(self)

        print self.model.data.qpos


    def _step(self, a):
        self.do_simulation(a, self.frame_skip)
        
        obs = self._get_obs()
        done = False
        
        contact_reward = 0.0
        grasp_reward = 0.0

        # Check for contact
        d = self.unwrapped.data
        for coni in range(d.ncon):
            con = d.obj.contact[coni]
            if con.geom1 != 0 and con.geom2 == 12:
                # Small box is touched but not by table
                contact_reward = 1.0

                box_height = self.model.data.qpos[11] - 0.025
                if box_height > 0:
                    grasp_reward = box_height * 100.0
        
        reward = contact_reward + grasp_reward

        return obs, reward, done, dict(reward_contact=contact_reward, reward_grasp=grasp_reward)

    def viewer_setup(self):
        # self.viewer.cam.trackbodyid = 0
        self.viewer.cam.distance = self.model.stat.extent * 2.5
        self.viewer.cam.elevation += 20


    def reset_model(self):
        c = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv,)
        )
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat,
            self.model.data.qvel.flat
        ])
