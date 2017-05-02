import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import mujoco_py.mjlib



class Box3dFixedReachEnvPixel(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'arm_claw_new.xml', 4)
        utils.EzPickle.__init__(self)
        
    def _step(self, a):
        self.do_simulation(a, self.frame_skip)
        
        obs = self._get_obs()
        done = False
        
        reach_reward = 0.0
        contact_reward = 0.0

        # Check for distance
        d = self.unwrapped.data
        # for coni in range(d.ncon):
        #     con = d.obj.contact[coni]
        #     if con.geom1 != 0 and con.geom2 == 12:
        #         # Small box is touched but not by table
        #         contact_reward = 1.0
        # print (d.site_xpos.flatten())
        distance = np.linalg.norm(d.site_xpos.flatten() - (list(d.qpos[-2:].flatten()) + [0.025]))
        if distance <= 0.4:
            reach_reward += 2.0 - distance*3
        reward = reach_reward

        return obs, reward, done, dict(reach_reward=reach_reward, contact_reward=contact_reward)


    def _get_obs(self):
        camera2_output = None
        self.camera2.render()
        data, width, height = self.camera2.get_image()
        camera2_output =  np.fromstring(data, dtype='uint8').reshape(height, width, 3)[::-1, :, :]
        return np.concatenate((self._render(mode="rgb_array"), camera2_output), axis = 2)

    def reset_model(self):
        c = 0.01
        qpos = [0.000, 3.133, 0.018, -1.500, -0.004, -0.000, 0.005, -0.001, 0.007]
        self.set_state(
            qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel,
        )
        return self._get_obs()
    
    def _get_viewer(self):
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer(init_width=500, init_height=500)
            self.viewer.start()
            self.viewer.set_model(self.model)
            self.viewer_setup()
        return self.viewer
    
    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = self.model.stat.extent * 2.5*1.4
        self.viewer.cam.elevation = -40
        self.viewer.cam.lookat[1]+=0.3
        
    def camera2_setup(self):
        self.camera2.cam.trackbodyid = -1
        self.camera2.cam.distance = self.model.stat.extent * 2.5*1.4
        self.camera2.cam.elevation = -30
        self.camera2.cam.lookat[0] += 0.4
        self.camera2.cam.lookat[1] += 0.3
        self.camera2.cam.lookat[2] += 0
        
        

class Box3dNoRewardEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'arm_claw_new.xml', 4)
        utils.EzPickle.__init__(self)
        

    def _step(self, a):
        self.do_simulation(a, self.frame_skip)
        
        obs = self._get_obs()
        done = False
        
        reach_reward = 0.0
        contact_reward = 0.0

        return obs, 0.0, done, dict(reach_reward=reach_reward, contact_reward=contact_reward)


    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat,
            self.model.data.qvel.flat
        ])

    def reset_model(self):
        c = 0.01
        qpos = [0.000, 3.133, 0.018, -1.500, -0.004, -0.000, 0.005, -0.001, 0.007]
        self.set_state(
            qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel,
        )
        return self._get_obs()

    def _get_obs(self):
        # return self.model.data.qpos.flat[:9]
        return np.concatenate([
           self.model.data.qpos.flat,
           self.model.data.qvel.flat
        ])

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = self.model.stat.extent * 2.5
        self.viewer.cam.elevation = -40



class Box3dReachEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'arm_claw.xml', 4)
        utils.EzPickle.__init__(self)
        
        self._randomize_box()


    def _randomize_box(self):
        state_noise = np.zeros(16)
        state_noise[9:11] = self.np_random.uniform(low=-0.1, high=0.1, size=2)
        self.set_state(
                self.model.data.qpos.ravel() + state_noise,
                self.model.data.qvel.ravel(),
        )


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
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = self.model.stat.extent * 2.5
        self.viewer.cam.elevation = -40


    def reset_model(self):
        c = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv,)
        )
        self._randomize_box()
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat,
            self.model.data.qvel.flat
        ])


class Box3dReachPosEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'arm_claw.xml', 4)
        utils.EzPickle.__init__(self)
        
        self._randomize_box()


    def _randomize_box(self):
        state_noise = np.zeros(16)
        state_noise[9:11] = self.np_random.uniform(low=-0.1, high=0.1, size=2)
        self.set_state(
                self.model.data.qpos.ravel() + state_noise,
                self.model.data.qvel.ravel(),
        )


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


    def reset_model(self):
        c = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv,)
        )
        self._randomize_box()
        return self._get_obs()

    def _get_obs(self):
        #return self.model.data.qpos.flat[:9]
        return np.concatenate([
            self.model.data.qpos.flat[:9],
            self.model.data.qvel.flat[:9]
        ])

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = self.model.stat.extent * 2.5
        self.viewer.cam.elevation = -40




class Box3dContactReachEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'arm_claw_new.xml', 4)
        utils.EzPickle.__init__(self)
        

    def _step(self, a):
        self.do_simulation(a, self.frame_skip)
        
        obs = self._get_obs()
        done = False
        
        reach_reward = 0.0
        contact_reward = 0.0

        # Check for distance
        d = self.unwrapped.data
        for coni in range(d.ncon):
            con = d.obj.contact[coni]
            if con.geom1 != 0 and con.geom2 == 12:
                # Small box is touched but not by table
                contact_reward = 1.0
        # print (d.site_xpos.flatten())
        # distance = np.linalg.norm(d.site_xpos.flatten() - (list(d.qpos[-2:].flatten()) + [0.025]))
        # if distance <= 0.4:
        #     reach_reward += 2.0 - distance*3

        reward = contact_reward + reach_reward

        return obs, reward, done, dict(reach_reward=reach_reward, contact_reward=contact_reward)


    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat,
            self.model.data.qvel.flat
        ])

    def reset_model(self):
        c = 0.01
        qpos = [0.000, 3.133, 0.018, -1.500, -0.004, -0.000, 0.005, -0.001, 0.007]
        self.set_state(
            qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel,
        )
        return self._get_obs()

    def _get_obs(self):
        # return self.model.data.qpos.flat[:9]
        return np.concatenate([
           self.model.data.qpos.flat,
           self.model.data.qvel.flat
        ])

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = self.model.stat.extent * 2.5
        self.viewer.cam.elevation = -40


class Box3dFixedReachEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'arm_claw_new.xml', 4)
        utils.EzPickle.__init__(self)
        

    def _step(self, a):
        self.do_simulation(a, self.frame_skip)
        
        obs = self._get_obs()
        done = False
        
        reach_reward = 0.0
        contact_reward = 0.0

        # Check for distance
        d = self.unwrapped.data
        # for coni in range(d.ncon):
        #     con = d.obj.contact[coni]
        #     if con.geom1 != 0 and con.geom2 == 12:
        #         # Small box is touched but not by table
        #         contact_reward = 1.0
        # print (d.site_xpos.flatten())
        distance = np.linalg.norm(d.site_xpos.flatten() - (list(d.qpos[-2:].flatten()) + [0.025]))
        if distance <= 0.4:
            reach_reward += 2.0 - distance*3

        reward = contact_reward + reach_reward

        return obs, reward, done, dict(reach_reward=reach_reward, contact_reward=contact_reward)


    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat,
            self.model.data.qvel.flat
        ])

    def reset_model(self):
        c = 0.01
        qpos = [0.000, 3.133, 0.018, -1.500, -0.004, -0.000, 0.005, -0.001, 0.007]
        self.set_state(
            qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel,
        )
        return self._get_obs()

    def _get_obs(self):
        # return self.model.data.qpos.flat[:9]
        return np.concatenate([
           self.model.data.qpos.flat,
           self.model.data.qvel.flat
        ])

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = self.model.stat.extent * 2.5
        self.viewer.cam.elevation = -40


class Box3dGraspEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'arm_claw.xml', 4)
        utils.EzPickle.__init__(self)


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
                box_height = self.model.data.qpos[11] - 0.025
                if box_height > 0.1:
                    grasp_reward = 1.0
        
        reward = contact_reward + grasp_reward

        return obs, reward, done, dict(reward_contact=contact_reward, reward_grasp=grasp_reward)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = self.model.stat.extent * 2.5
        self.viewer.cam.elevation = -40


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
