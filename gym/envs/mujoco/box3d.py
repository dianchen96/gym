import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.spaces.box import Box
import mujoco_py.mjlib
import math

class Box3dReachTableEnv4StepWithLimit(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'arm_reach_one_box_contact_free_with_limit.xml', 4)
        utils.EzPickle.__init__(self)
        

    def _step(self, a):
        for _ in range(4):
            self.do_simulation(a, self.frame_skip)
        
        obs = self._get_obs()

        done = False
        
        
        d = self.unwrapped.data
        distance = np.linalg.norm(d.site_xpos.flatten() - list(d.qpos[4:7].flatten()))
        reach_reward = -distance
        # import pdb; pdb.set_trace()
        #push reward is high if the box is moving in a certain direction
        box_vel = d.qvel[4:7]
        target_dir = [0,1,0]
        push_reward = np.dot(box_vel.flatten(), target_dir)
        push_reward = -np.abs(0.1 - push_reward)

        reward =  reach_reward
        # print(reward)
        return obs, reward, done, {}

    def reset_model(self):

        qpos = self.init_qpos.copy()
        qpos[0]+=(np.random.rand()*(2.27)-1.57)
        qpos[1]+=(np.random.rand()*2.42 - 0.85)
        qpos[2]+=(np.random.rand()*1.7 - 0.85)
        qpos[3]+=(np.random.rand()*2.55 - 1.5)
        qpos[5]+=0.1
        qpos[4:6] += (np.random.rand(2)*0.7 - 0.35)
        self.set_state(qpos, self.init_qvel)
        
        return self._get_obs()

    def _get_obs(self):
        # return self.model.data.qpos.flat[:9]
        # import pdb; pdb.set_trace()
        return np.concatenate([
           self.model.data.qpos.flat,
           self.model.data.qvel.flat
        ])

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = self.model.stat.extent * 4.0
        self.viewer.cam.elevation = -40
class Box3dReachTableEnv4Step(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'arm_reach_one_box_contact_free.xml', 4)
        utils.EzPickle.__init__(self)
        

    def _step(self, a):
        for _ in range(4):
            self.do_simulation(a, self.frame_skip)
        
        obs = self._get_obs()

        done = False
        
        
        d = self.unwrapped.data
        distance = np.linalg.norm(d.site_xpos.flatten() - list(d.qpos[4:7].flatten()))
        reach_reward = -distance
        # import pdb; pdb.set_trace()
        #push reward is high if the box is moving in a certain direction
        box_vel = d.qvel[4:7]
        target_dir = [0,1,0]
        push_reward = np.dot(box_vel.flatten(), target_dir)
        push_reward = -np.abs(0.1 - push_reward)

        reward =  reach_reward
        return obs, reward, done, {}

    def reset_model(self):

        qpos = self.init_qpos.copy()
        qpos[:4] = np.random.rand(4)*6.28
        qpos[5]+=0.1
        qpos[4:6] += (np.random.rand(2)*0.7 - 0.35)
        self.set_state(qpos, self.init_qvel)
        
        return self._get_obs()

    def _get_obs(self):
        # return self.model.data.qpos.flat[:9]
        # import pdb; pdb.set_trace()
        return np.concatenate([
           self.model.data.qpos.flat,
           self.model.data.qvel.flat
        ])

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = self.model.stat.extent * 4.0
        self.viewer.cam.elevation = -40

class Box3dFixPushingEnv4StepReal(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.goal = np.array([0.55,0.1])
        mujoco_env.MujocoEnv.__init__(self, 'arm_reach_one_real.xml', 4)
        utils.EzPickle.__init__(self)
    def check_contact(self):
        d = self.unwrapped.data
        for coni in range(d.ncon):
            con = d.obj.contact[coni]
            if  (con.geom1 != 0) \
            and (con.geom2 == 3):
                # Small box is touched but not by table
                return True
        return False
        
    def _step(self, a):
        for _ in range(4):
            self.do_simulation(a, self.frame_skip)
        
        obs = self._get_obs()

        done = False
        
        
        d = self.unwrapped.data
        distance = np.linalg.norm(d.site_xpos.flatten() - list(d.qpos[4:7].flatten()))
        reach_reward = -distance
    
        box_pos_x = d.qpos[4:6].flatten()
        push_reward = -np.sum(np.square(box_pos_x - self.goal))*100
        reward = push_reward
        print(self.check_contact())
        return obs, reward, done, {'contact':self.check_contact()}
    def reset_model(self):

        qpos = self.init_qpos.copy()
        qpos[0]+=(np.random.rand()*(2.27)-1.57)
        qpos[1]+=(np.random.rand()*2.42 - 0.85)
        qpos[2]+=(np.random.rand()*1.7 - 0.85)
        qpos[3]+=(np.random.rand()*2.55 - 1.5)
        # qpos[4] += 0.5
        self.set_state(qpos, self.init_qvel)
    
        return self._get_obs()

    def _get_obs(self):
        qpos = self.model.data.qpos.flat.copy()
        qvel = self.model.data.qvel.flat.copy()
        return np.concatenate([
            qpos,
            qvel
        ])

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = self.model.stat.extent * 4.0
        self.viewer.cam.elevation = -40

class Box3dRescaledFixPushingEnv4StepReal3(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.goal = np.array([0.55,0.1, 0.2])
        mujoco_env.MujocoEnv.__init__(self, 'arm_reach_one_real3.xml', 4)
        utils.EzPickle.__init__(self)
    def check_contact(self):
        d = self.unwrapped.data
        for coni in range(d.ncon):
            con = d.obj.contact[coni]
            if  (con.geom1 != 0) \
            and (con.geom2 == 3):
                # Small box is touched but not by table
                return True
        return False
        
    def _step(self, a):
        for _ in range(4):
            self.do_simulation(a, self.frame_skip)
        
        obs = self._get_obs()

        done = False
        
        
        d = self.unwrapped.data
        #reward to ensure that end_effect box goal are in a line
        box2hand = d.qpos[4:7] - d.site_xpos.flatten()
        goal2box = self.goal - d.qpos[4:7]
        
        import pdb; pdb.set_trace()
        aline_
        distance = np.linalg.norm(d.site_xpos.flatten() - list(d.qpos[4:7].flatten()))
        reach_reward = -distance
    
        box_pos_x = d.qpos[4:7].flatten()
        push_reward = -np.sum(np.square(box_pos_x - self.goal))*100
        reward = push_reward
        # print(self.check_contact())
        return obs, reward, done, {'contact':self.check_contact()}
    def reset_model(self):

        qpos = self.init_qpos.copy()
        qpos[0]+=(np.random.rand()*(2.27)-1.57)
        qpos[1]+=(np.random.rand()*2.42 - 0.85)
        qpos[2]+=(np.random.rand()*1.7 - 0.85)
        qpos[3]+=(np.random.rand()*2.55 - 1.5)
        # qpos[4] += 0.5
        self.set_state(qpos, self.init_qvel)
    
        return self._get_obs()

    def _get_obs(self):
        
        qpos = self.model.data.qpos.flat.copy()
        qvel = self.model.data.qvel.flat.copy()
        qpos[4:7] = qpos[4:7]*10
        return np.concatenate([
            qpos,
            qvel
        ])

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = self.model.stat.extent * 4.0
        self.viewer.cam.elevation = -40

class Box3dRescaledFixPushingEnv4StepReal2(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.goal = np.array([0.55,0.1])
        mujoco_env.MujocoEnv.__init__(self, 'arm_reach_one_real2.xml', 4)
        utils.EzPickle.__init__(self)
    def check_contact(self):
        d = self.unwrapped.data
        for coni in range(d.ncon):
            con = d.obj.contact[coni]
            if  (con.geom1 != 0) \
            and (con.geom2 == 3):
                # Small box is touched but not by table
                return True
        return False
        
    def _step(self, a):
        for _ in range(4):
            self.do_simulation(a, self.frame_skip)
        
        obs = self._get_obs()

        done = False
        d = self.unwrapped.data
        

    
        box_pos_x = d.qpos[4:6].flatten()
        push_reward = -np.sum(np.square(box_pos_x - self.goal))*100
        reward = push_reward
        # print(self.check_contact())
        return obs, reward, done, {'contact':self.check_contact()}
    def reset_model(self):

        qpos = self.init_qpos.copy()
        qpos[0]+=(np.random.rand()*(2.27)-1.57)
        qpos[1]+=(np.random.rand()*2.42 - 0.85)
        qpos[2]+=(np.random.rand()*1.7 - 0.85)
        qpos[3]+=(np.random.rand()*2.55 - 1.5)
        # qpos[4] += 0.5
        self.set_state(qpos, self.init_qvel)
    
        return self._get_obs()

    def _get_obs(self):
        qpos = self.model.data.qpos.flat.copy()
        qvel = self.model.data.qvel.flat.copy()
        qpos[4:6] = qpos[4:6]*10
        return np.concatenate([
            qpos,
            qvel
        ])

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = self.model.stat.extent * 4.0
        self.viewer.cam.elevation = -40

#Box3dPush-v21 weaker arm, heavier box and 0.4 friction coefficent
class Box3dRescaledFixPushingEnv4StepReal1(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.goal = np.array([0.55,0.1, 0.2])
        mujoco_env.MujocoEnv.__init__(self, 'arm_reach_one_real1.xml', 4)
        utils.EzPickle.__init__(self)
    def check_contact(self):
        d = self.unwrapped.data
        for coni in range(d.ncon):
            con = d.obj.contact[coni]
            if  (con.geom1 != 0) \
            and (con.geom2 == 3):
                # Small box is touched but not by table
                return True
        return False
        
    # def _step(self, a):
    #     for _ in range(4):
    #         self.do_simulation(a, self.frame_skip)
        
    #     obs = self._get_obs()

    #     done = False
        
        
    #     d = self.unwrapped.data
    #     arm2box = d.qpos[4:7].flatten() - d.site_xpos.flatten()
    #     goal2box = self.goal - d.qpos[4:7].flatten()
    #     aline_reward = np.dot(arm2box.T, goal2box)/np.linalg.norm(arm2box)/np.linalg.norm(goal2box)
    #     reach_reward = 0
    #     if np.linalg.norm(arm2box) < 0.07:
    #         reach_reward = 0.4
    #     # import pdb; pdb.set_trace()
    #     # print(alipoline_reward)
    #     box_pos = d.qpos[4:7].flatten()
    #     push_reward = -np.sum(np.square(box_pos - self.goal))*100
    #     reward = push_reward + aline_reward + reach_reward
        # print(self.check_contact())
    def _step(self, a):
        for _ in range(4):
            self.do_simulation(a, self.frame_skip)
        
        obs = self._get_obs()

        done = False
        
        
        d = self.unwrapped.data
        action_penalty = np.sum(np.square(a))
        box_pos_x = d.qpos[4:7].flatten()
        push_reward = -np.sum(np.square(box_pos_x - self.goal))*10
        reward = push_reward - action_penalty*0.01
        return obs, reward, done, {'contact':self.check_contact()}
    def reset_model(self):

        qpos = self.init_qpos.copy()
        qpos[0]+=(np.random.rand()*(2.27)-1.57)
        qpos[1]+=(np.random.rand()*2.42 - 0.85)
        qpos[2]+=(np.random.rand()*1.7 - 0.85)
        qpos[3]+=(np.random.rand()*2.55 - 1.5)
        # qpos[4] += 0.5
        self.set_state(qpos, self.init_qvel)
    
        return self._get_obs()

    def _get_obs(self):
        # position and velocity all relate to the arm end effect
        # import pdb; pdb.set_trace()
        qpos = self.model.data.qpos.flat.copy()
        qvel = self.model.data.qvel.flat.copy()
        
        qpos[4:7] = qpos[4:7]*10
        return np.concatenate([
            qpos,
            qvel
        ])

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = self.model.stat.extent * 4.0
        self.viewer.cam.elevation = -40
        
#box3dpush-v2 the only good policy we have right now
class Box3dRescaledFixPushingEnv4StepReal(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.goal = np.array([0.55,0.1])
        mujoco_env.MujocoEnv.__init__(self, 'arm_reach_one_real.xml', 4)
        utils.EzPickle.__init__(self)
        self.last_box_pos = None
        self.last_site_pos = None
    def get_invariant_obs(self):
        d = self.unwrapped.data
        site_xpos = d.site_xpos.flatten()
        qpos = self.model.data.qpos.flat.copy()
        qvel = self.model.data.qvel.flat.copy()
        site_xvel = site_xpos - self.last_site_pos
        box_xvel = qpos[4:7] - self.last_box_pos
        box_vel2site_vel = box_xvel - site_xvel
        arm2box = qpos[4:7] - site_xpos
        qpos[4:7] = arm2box * 10
        qvel[4:7] = box_vel2site_vel
        #box orientation should be fine...
        #Also add site xpos to observation
        return np.concatenate([qpos, qvel, site_xpos])
        
    def check_contact(self):
        d = self.unwrapped.data
        for coni in range(d.ncon):
            con = d.obj.contact[coni]
            if  (con.geom1 != 0) \
            and (con.geom2 == 3):
                # Small box is touched but not by table
                return True
        return False
        
    def _step(self, a):
        for _ in range(4):
            self.do_simulation(a, self.frame_skip)
        
        obs = self._get_obs()

        done = False
        
        
        d = self.unwrapped.data
        distance = np.linalg.norm(d.site_xpos.flatten() - list(d.qpos[4:7].flatten()))
        reach_reward = -distance
    
        box_pos_x = d.qpos[4:6].flatten()
        push_reward = -np.sum(np.square(box_pos_x - self.goal))*100
        reward = push_reward
        # print(self.check_contact())
        return obs, reward, done, {'contact':self.check_contact()}
    def reset_model(self):

        qpos = self.init_qpos.copy()
        qpos[0]+=(np.random.rand()*(2.27)-1.57)
        qpos[1]+=(np.random.rand()*2.42 - 0.85)
        qpos[2]+=(np.random.rand()*1.7 - 0.85)
        qpos[3]+=(np.random.rand()*2.55 - 1.5)
        # qpos[4] += 0.5
        self.last_box_pos = qpos[4:7]
        self.set_state(qpos, self.init_qvel)
        self.last_site_pos = self.unwrapped.data.site_xpos.flatten()
        return self._get_obs()

    def _get_obs(self):
        qpos = self.model.data.qpos.flat.copy()
        qvel = self.model.data.qvel.flat.copy()
        qpos[4:7] = qpos[4:7]*10
        return np.concatenate([
            qpos,
            qvel
        ])

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = self.model.stat.extent * 4.0
        self.viewer.cam.elevation = -40
        
"""
Box is fixed, but the push direction changes. 
"""
class Box3dRescaledPushingEnv4StepReal(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.goal = None
        self.angle_list = [2*np.pi/20*i for i in range(20)]
        angle = self.angle_list[np.random.randint(20)]
        self.goal = np.array([0.4*np.cos(angle), 0.4*np.sin(angle)])
        
        mujoco_env.MujocoEnv.__init__(self, 'arm_reach_one_real.xml', 4)
        utils.EzPickle.__init__(self)
    def check_contact(self):
        d = self.unwrapped.data
        for coni in range(d.ncon):
            con = d.obj.contact[coni]
            if  (con.geom1 != 0) \
            and (con.geom2 == 3):
                # Small box is touched but not by table
                return True
        return False
        
    def _step(self, a):
        for _ in range(4):
            self.do_simulation(a, self.frame_skip)
        
        obs = self._get_obs()

        done = False
        
        
        d = self.unwrapped.data
        # distance = np.linalg.norm(d.site_xpos.flatten() - list(d.qpos[4:7].flatten()))
        # reach_reward = -distance
    
        box_pos_x = d.qpos[4:6].flatten()
        push_reward = -np.sum(np.square(box_pos_x - self.goal))*100
        
        if self.check_contact():
            push_reward += 0.2
        reward = push_reward
        return obs, reward, done, {'contact':self.check_contact()}
    
    def reset_model(self):
        qpos = self.init_qpos.copy()
        qpos[0]+=(np.random.rand()*(2.27)-1.57)
        qpos[1]+=(np.random.rand()*2.42 - 0.85)
        qpos[2]+=(np.random.rand()*1.7 - 0.85)
        qpos[3]+=(np.random.rand()*2.55 - 1.5)
        qpos[4] += 0.12
        qpos[5] += 0.1
        import pdb; pdb.set_trace()
        #randomly select a goal, push distance always 0.3
        angle = self.angle_list[np.random.randint(20)]
        self.goal = np.array([0.4*np.cos(angle), 0.4*np.sin(angle)])
        self.set_state(qpos, self.init_qvel)
        return self._get_obs()

    def _get_obs(self):
        assert self.goal is not None
        qpos = self.model.data.qpos.flat.copy()
        qvel = self.model.data.qvel.flat.copy()
        qpos[4:7] = qpos[4:7]*10
        d = self.unwrapped.data
        vector = d.site_xpos.flatten() - d.qpos[4:7].flatten()
        return np.concatenate([
            qpos,
            qvel,
            self.goal,
            vector,
        ])

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = self.model.stat.extent * 4.0
        self.viewer.cam.elevation = -40
        
        
class Box3dRescaledPushingEnv4Step(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.goal = np.array([0,0])
        mujoco_env.MujocoEnv.__init__(self, 'arm_reach_one.xml', 4)
        utils.EzPickle.__init__(self)
        
    def _step(self, a):
        for _ in range(4):
            self.do_simulation(a, self.frame_skip)
        
        obs = self._get_obs()

        done = False
        
        
        d = self.unwrapped.data
        distance = np.linalg.norm(d.site_xpos.flatten() - list(d.qpos[4:7].flatten()))
        reach_reward = -distance
    
        box_pos_x = d.qpos[4:6]
        push_reward = -np.sum(np.square(box_pos_x - self.goal))*100

        reward = push_reward

        return obs, reward, done, {}

    def reset_model(self):

        qpos = self.init_qpos.copy()
        qpos[0]+=(np.random.rand()*(2.27)-1.57)
        qpos[1]+=(np.random.rand()*2.42 - 0.85)
        qpos[2]+=(np.random.rand()*1.7 - 0.85)
        qpos[3]+=(np.random.rand()*2.55 - 1.5)
        qpos[5]+=0.1
        qpos[4:6] += (np.random.rand(2)*0.7 - 0.35)
        self.set_state(qpos, self.init_qvel)
        
        return self._get_obs()

    def _get_obs(self):
        qpos = self.model.data.qpos.flat.copy()
        qvel = self.model.data.qvel.flat.copy()
        qpos[4:7] = qpos[4:7]*10
        return np.concatenate([
            qpos,
            qvel
        ])

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = self.model.stat.extent * 4.0
        self.viewer.cam.elevation = -40

class Box3dFixedPushingEnv4Step(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'arm_reach_one.xml', 4)
        utils.EzPickle.__init__(self)
        

    def _step(self, a):
        for _ in range(4):
            self.do_simulation(a, self.frame_skip)
        
        obs = self._get_obs()

        done = False
        
        
        d = self.unwrapped.data
        distance = np.linalg.norm(d.site_xpos.flatten() - list(d.qpos[4:7].flatten()))
        reach_reward = -distance
        # import pdb; pdb.set_trace()
        #push reward is high if the box is moving in a certain direction
        box_vel = d.qvel[4:7]
        target_dir = [0,1,0]
        push_reward = np.dot(box_vel.flatten(), target_dir)
        push_reward = -np.abs(0.1 - push_reward)

        reward =  reach_reward

        return obs, reward, done, {}

    def reset_model(self):

        qpos = self.init_qpos.copy()
        qpos[0]+=(np.random.rand()*(2.27)-1.57)
        qpos[1]+=(np.random.rand()*2.42 - 0.85)
        qpos[2]+=(np.random.rand()*1.7 - 0.85)
        qpos[3]+=(np.random.rand()*2.55 - 1.5)
        qpos[5]+=0.1
        qpos[4:6] += (np.random.rand(2)*0.7 - 0.35)
        self.set_state(qpos, self.init_qvel)
        
        return self._get_obs()

    def _get_obs(self):
        # return self.model.data.qpos.flat[:9]
        # import pdb; pdb.set_trace()
        return np.concatenate([
           self.model.data.qpos.flat,
           self.model.data.qvel.flat
        ])

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = self.model.stat.extent * 4.0
        self.viewer.cam.elevation = -40

class Box3dFixedReachSixBoxEnvMulContactTwoCamNoBoxVel(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'arm_reach_6_boxes_big.xml', 4)
        utils.EzPickle.__init__(self)
        
    def check_contact(self):
        d = self.unwrapped.data
        for coni in range(d.ncon):
            con = d.obj.contact[coni]
            if  (con.geom1 != 0 and \
                 con.geom1 != 3 and \
                 con.geom1 != 4 and \
                 con.geom1 != 5 and \
                 con.geom1 != 6 and \
                 con.geom1 != 7 and \
                 con.geom1 != 8) \
            and (con.geom2 == 3 or \
                 con.geom2 == 4 or \
                 con.geom2 == 5 or \
                 con.geom2 == 6 or \
                 con.geom2 == 7 or \
                 con.geom2 == 8):
                # Small box is touched but not by table
                return True
        return False

    def check_obj_contact(self):
        d = self.unwrapped.data
        for coni in range(d.ncon):
            con = d.obj.contact[coni]
            if  (con.geom1 == 3 or \
                 con.geom1 == 4 or \
                 con.geom1 == 5 or \
                 con.geom1 == 6 or \
                 con.geom1 == 7 or \
                 con.geom1 == 8) \
            and (con.geom2 == 3 or \
                 con.geom2 == 4 or \
                 con.geom2 == 5 or \
                 con.geom2 == 6 or \
                 con.geom2 == 7 or \
                 con.geom2 == 8):
                # Small box is touched but not by table
                return True
        return False

    def check_table_contact(self):
        d = self.unwrapped.data
        for coni in range(d.ncon):
            con = d.obj.contact[coni]
            if  (con.geom1 == 0) \
            and (con.geom2 == 1 or con.geom2 == 2):
                # Small box is touched but not by table
                return True
        return False

    def _step(self, a):

        self.do_simulation(a, self.frame_skip)
        done = False
        obs = self._get_obs()
        
        contact_reward = 0.0
        table_contact_reward = 0.0

        if self.check_contact():
            contact_reward += 1.0
        if self.check_table_contact():
            table_contact_reward += 1.0

        reward = contact_reward

        return obs, reward, done, dict(contact_reward=contact_reward, table_contact_reward=table_contact_reward)

    def _get_obs(self):
        # return self.model.data.qpos.flat[:9]
        return np.concatenate([
           self.model.data.qpos.flat,
           self.model.data.qvel.flat[:4]
        ])

    def reset_model(self):
        c = 0.5
        # qpos = [0.000, 3.133, 0.018, -1.500, -0.004, -0.000, 0.005, -0.001, 0.007]
        while True:
            qpos = self.init_qpos.copy()
            qpos[:4] = [0.000, 1.000, 0.000, -0.004] + self.np_random.uniform(low=-0.1, high=0.1, size=4)
            qpos[4:6] += self.np_random.uniform(low=-c, high=c, size=2)
            qpos[11:13] += self.np_random.uniform(low=-c, high=c, size=2)
            qpos[18:20] += self.np_random.uniform(low=-c, high=c, size=2)
            qpos[25:27] += self.np_random.uniform(low=-c, high=c, size=2)
            qpos[32:34] += self.np_random.uniform(low=-c, high=c, size=2)
            qpos[39:41] += self.np_random.uniform(low=-c, high=c, size=2)


            self.set_state(
                np.array(qpos),
                self.init_qvel,
            )

            if not self.check_obj_contact():
                break
                
        obs = self._get_obs()
        return obs
    
    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = self.model.stat.extent * 3.0
        self.viewer.cam.elevation = -40
        self.viewer.cam.lookat[0] += 0.2
        self.viewer.cam.lookat[1] -= 0.1


class Box3dFixedReachSixBoxEnvMulContactTwoCam10Step(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'arm_reach_6_boxes_big.xml', 4)
        utils.EzPickle.__init__(self)
        
    def check_contact(self):
        d = self.unwrapped.data
        for coni in range(d.ncon):
            con = d.obj.contact[coni]
            if  (con.geom1 != 0 and \
                 con.geom1 != 3 and \
                 con.geom1 != 4 and \
                 con.geom1 != 5 and \
                 con.geom1 != 6 and \
                 con.geom1 != 7 and \
                 con.geom1 != 8) \
            and (con.geom2 == 3 or \
                 con.geom2 == 4 or \
                 con.geom2 == 5 or \
                 con.geom2 == 6 or \
                 con.geom2 == 7 or \
                 con.geom2 == 8):
                # Small box is touched but not by table
                return True
        return False

    def check_obj_contact(self):
        d = self.unwrapped.data
        for coni in range(d.ncon):
            con = d.obj.contact[coni]
            if  (con.geom1 == 3 or \
                 con.geom1 == 4 or \
                 con.geom1 == 5 or \
                 con.geom1 == 6 or \
                 con.geom1 == 7 or \
                 con.geom1 == 8) \
            and (con.geom2 == 3 or \
                 con.geom2 == 4 or \
                 con.geom2 == 5 or \
                 con.geom2 == 6 or \
                 con.geom2 == 7 or \
                 con.geom2 == 8):
                # Small box is touched but not by table
                return True
        return False

    def check_table_contact(self):
        d = self.unwrapped.data
        for coni in range(d.ncon):
            con = d.obj.contact[coni]
            if  (con.geom1 == 0) \
            and (con.geom2 == 1 or con.geom2 == 2):
                # Small box is touched but not by table
                return True
        return False

    def _step(self, a):
        for _ in range(10):
            self.do_simulation(a, self.frame_skip)
        done = False
        obs = self._get_obs()
        
        contact_reward = 0.0
        table_contact_reward = 0.0

        if self.check_contact():
            contact_reward += 1.0
        if self.check_table_contact():
            table_contact_reward += 1.0

        reward = contact_reward

        return obs, reward, done, dict(contact_reward=contact_reward, table_contact_reward=table_contact_reward)

    def _get_obs(self):
        # return self.model.data.qpos.flat[:9]
        return np.concatenate([
           self.model.data.qpos.flat,
           self.model.data.qvel.flat
        ])

    def reset_model(self):
        c = 0.5
        # qpos = [0.000, 3.133, 0.018, -1.500, -0.004, -0.000, 0.005, -0.001, 0.007]
        while True:
            qpos = self.init_qpos.copy()
            qpos[:4] = [0.000, 1.000, 0.000, -0.004] + self.np_random.uniform(low=-0.1, high=0.1, size=4)
            qpos[4:6] += self.np_random.uniform(low=-c, high=c, size=2)
            qpos[11:13] += self.np_random.uniform(low=-c, high=c, size=2)
            qpos[18:20] += self.np_random.uniform(low=-c, high=c, size=2)
            qpos[25:27] += self.np_random.uniform(low=-c, high=c, size=2)
            qpos[32:34] += self.np_random.uniform(low=-c, high=c, size=2)
            qpos[39:41] += self.np_random.uniform(low=-c, high=c, size=2)


            self.set_state(
                np.array(qpos),
                self.init_qvel,
            )

            if not self.check_obj_contact():
                break
                
        obs = self._get_obs()
        return obs
    
    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = self.model.stat.extent * 3.0
        self.viewer.cam.elevation = -40
        self.viewer.cam.lookat[0] += 0.2
        self.viewer.cam.lookat[1] -= 0.1


class Box3dFixedReachSixBoxEnvMulContactTwoCam4Step(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'arm_reach_6_boxes_big.xml', 4)
        utils.EzPickle.__init__(self)
        
    def check_contact(self):
        d = self.unwrapped.data
        for coni in range(d.ncon):
            con = d.obj.contact[coni]
            if  (con.geom1 != 0 and \
                 con.geom1 != 3 and \
                 con.geom1 != 4 and \
                 con.geom1 != 5 and \
                 con.geom1 != 6 and \
                 con.geom1 != 7 and \
                 con.geom1 != 8) \
            and (con.geom2 == 3 or \
                 con.geom2 == 4 or \
                 con.geom2 == 5 or \
                 con.geom2 == 6 or \
                 con.geom2 == 7 or \
                 con.geom2 == 8):
                # Small box is touched but not by table
                return True
        return False

    def check_obj_contact(self):
        d = self.unwrapped.data
        for coni in range(d.ncon):
            con = d.obj.contact[coni]
            if  (con.geom1 == 3 or \
                 con.geom1 == 4 or \
                 con.geom1 == 5 or \
                 con.geom1 == 6 or \
                 con.geom1 == 7 or \
                 con.geom1 == 8) \
            and (con.geom2 == 3 or \
                 con.geom2 == 4 or \
                 con.geom2 == 5 or \
                 con.geom2 == 6 or \
                 con.geom2 == 7 or \
                 con.geom2 == 8):
                # Small box is touched but not by table
                return True
        return False

    def check_table_contact(self):
        d = self.unwrapped.data
        for coni in range(d.ncon):
            con = d.obj.contact[coni]
            if  (con.geom1 == 0) \
            and (con.geom2 == 1 or con.geom2 == 2):
                # Small box is touched but not by table
                return True
        return False

    def _step(self, a):
        for _ in range(4):
            self.do_simulation(a, self.frame_skip)
        done = False
        obs = self._get_obs()
        
        contact_reward = 0.0
        table_contact_reward = 0.0

        if self.check_contact():
            contact_reward += 1.0
        if self.check_table_contact():
            table_contact_reward += 1.0

        reward = contact_reward

        return obs, reward, done, dict(contact_reward=contact_reward, table_contact_reward=table_contact_reward)

    def _get_obs(self):
        # return self.model.data.qpos.flat[:9]
        return np.concatenate([
           self.model.data.qpos.flat,
           self.model.data.qvel.flat
        ])

    def reset_model(self):
        c = 0.5
        # qpos = [0.000, 3.133, 0.018, -1.500, -0.004, -0.000, 0.005, -0.001, 0.007]
        while True:
            qpos = self.init_qpos.copy()
            qpos[:4] = [0.000, 1.000, 0.000, -0.004] + self.np_random.uniform(low=-0.1, high=0.1, size=4)
            qpos[4:6] += self.np_random.uniform(low=-c, high=c, size=2)
            qpos[11:13] += self.np_random.uniform(low=-c, high=c, size=2)
            qpos[18:20] += self.np_random.uniform(low=-c, high=c, size=2)
            qpos[25:27] += self.np_random.uniform(low=-c, high=c, size=2)
            qpos[32:34] += self.np_random.uniform(low=-c, high=c, size=2)
            qpos[39:41] += self.np_random.uniform(low=-c, high=c, size=2)


            self.set_state(
                np.array(qpos),
                self.init_qvel,
            )

            if not self.check_obj_contact():
                break
                
        obs = self._get_obs()
        return obs
    
    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = self.model.stat.extent * 3.0
        self.viewer.cam.elevation = -40
        self.viewer.cam.lookat[0] += 0.2
        self.viewer.cam.lookat[1] -= 0.1        


class Box3dFixedReachSixBoxEnvMulContactTwoCam(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'arm_reach_6_boxes_big.xml', 4)
        utils.EzPickle.__init__(self)
        
    def check_contact(self):
        d = self.unwrapped.data
        for coni in range(d.ncon):
            con = d.obj.contact[coni]
            if  (con.geom1 != 0 and \
                 con.geom1 != 3 and \
                 con.geom1 != 4 and \
                 con.geom1 != 5 and \
                 con.geom1 != 6 and \
                 con.geom1 != 7 and \
                 con.geom1 != 8) \
            and (con.geom2 == 3 or \
                 con.geom2 == 4 or \
                 con.geom2 == 5 or \
                 con.geom2 == 6 or \
                 con.geom2 == 7 or \
                 con.geom2 == 8):
                # Small box is touched but not by table
                return True
        return False

    def check_obj_contact(self):
        d = self.unwrapped.data
        for coni in range(d.ncon):
            con = d.obj.contact[coni]
            if  (con.geom1 == 3 or \
                 con.geom1 == 4 or \
                 con.geom1 == 5 or \
                 con.geom1 == 6 or \
                 con.geom1 == 7 or \
                 con.geom1 == 8) \
            and (con.geom2 == 3 or \
                 con.geom2 == 4 or \
                 con.geom2 == 5 or \
                 con.geom2 == 6 or \
                 con.geom2 == 7 or \
                 con.geom2 == 8):
                # Small box is touched but not by table
                return True
        return False

    def check_table_contact(self):
        d = self.unwrapped.data
        for coni in range(d.ncon):
            con = d.obj.contact[coni]
            if  (con.geom1 == 0) \
            and (con.geom2 == 1 or con.geom2 == 2):
                # Small box is touched but not by table
                return True
        return False

    def _step(self, a):

        self.do_simulation(a, self.frame_skip)
        done = False
        obs = self._get_obs()
        
        contact_reward = 0.0
        table_contact_reward = 0.0

        if self.check_contact():
            contact_reward += 1.0
        if self.check_table_contact():
            table_contact_reward += 1.0

        reward = contact_reward

        return obs, reward, done, dict(contact_reward=contact_reward, table_contact_reward=table_contact_reward)

    def _get_obs(self):
        # return self.model.data.qpos.flat[:9]
        return np.concatenate([
           self.model.data.qpos.flat,
           self.model.data.qvel.flat
        ])

    def reset_model(self):
        c = 0.5
        # qpos = [0.000, 3.133, 0.018, -1.500, -0.004, -0.000, 0.005, -0.001, 0.007]
        while True:
            qpos = self.init_qpos.copy()
            qpos[:4] = [0.000, 1.000, 0.000, -0.004] + self.np_random.uniform(low=-0.1, high=0.1, size=4)
            qpos[4:6] += self.np_random.uniform(low=-c, high=c, size=2)
            qpos[11:13] += self.np_random.uniform(low=-c, high=c, size=2)
            qpos[18:20] += self.np_random.uniform(low=-c, high=c, size=2)
            qpos[25:27] += self.np_random.uniform(low=-c, high=c, size=2)
            qpos[32:34] += self.np_random.uniform(low=-c, high=c, size=2)
            qpos[39:41] += self.np_random.uniform(low=-c, high=c, size=2)


            self.set_state(
                np.array(qpos),
                self.init_qvel,
            )

            if not self.check_obj_contact():
                break
                
        obs = self._get_obs()
        return obs
    
    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = self.model.stat.extent * 3.0
        self.viewer.cam.elevation = -40
        self.viewer.cam.lookat[0] += 0.2
        self.viewer.cam.lookat[1] -= 0.1



class Box3dFixedReachPixelMulObjConAvoidEnvOneEasy(mujoco_env.MujocoPixel2CamEnv, utils.EzPickle):
    def __init__(self, width=84, height=84, num_step=2):
        self.num_step = num_step
        mujoco_env.MujocoPixel2CamEnv.__init__(self, 'arm_reach_one.xml', 4, width, height, "grey")
        utils.EzPickle.__init__(self)
        self.observation_space = Box(low=0, high=255, shape=(width, height, 2 * num_step))
        
    def check_contact(self):
        d = self.unwrapped.data
        for coni in range(d.ncon):
            con = d.obj.contact[coni]
            if  (con.geom1 != 0) \
            and (con.geom2 == 3):
                # Small box is touched but not by table
                return True
        return False

    def check_obj_contact(self):
        return False

    def check_table_contact(self):
        d = self.unwrapped.data
        for coni in range(d.ncon):
            con = d.obj.contact[coni]
            if  (con.geom1 == 0) \
            and (con.geom2 == 1 or con.geom2 == 2):
                # Small box is touched but not by table
                return True
        return False

    def _step(self, a):
        obses = []
        for _ in range(self.num_step):
            self.do_simulation(a, self.frame_skip)
            obses.append(self._get_obs())

        done = False
        obs = np.concatenate(obses, axis=2)
        
        reach_reward = 0.0
        contact_reward = 0.0
        table_contact_reward = 0.0

        if self.check_contact():
            contact_reward += 1.0
        if self.check_table_contact():
            table_contact_reward += 1.0

        # Check for distance
        d = self.unwrapped.data
        distance = np.linalg.norm(d.site_xpos.flatten() - list(d.qpos[4:7].flatten()))
        if distance <= 0.4:
            reach_reward += 2.0 - distance*3

        reward = reach_reward

        return obs, reward, done, dict(contact_reward=contact_reward, table_contact_reward=table_contact_reward)


    def reset_model(self):
        c = 0.2
        # qpos = [0.000, 3.133, 0.018, -1.500, -0.004, -0.000, 0.005, -0.001, 0.007]
        while True:
            qpos = self.init_qpos.copy()

            qpos[:4] = [2.94179085e-02, 3.17722328e+00, -2.95601665e-01, -1.56731661e+00]

            theta = np.random.uniform(low=0.0, high=2 * math.pi, size=1)
            offset = 0.4 + np.random.uniform(low=-c, high=c, size=1)
            qpos[4] += offset * math.cos(theta)
            qpos[5] += offset * math.sin(theta)

            self.set_state(
                np.array(qpos),
                self.init_qvel,
            )

            if not self.check_obj_contact():
                break
                
        obs = self._get_obs()
        return np.concatenate([obs for _ in range(self.num_step)], axis=2)
    
    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = self.model.stat.extent * 3.5
        self.viewer.cam.elevation = -50
        self.viewer.cam.lookat[0] += 0.1
        self.viewer.cam.lookat[1] -= 0.1
        # self.viewer.cam.lookat[2] += 0.2



    def camera2_setup(self):
        self.camera2.cam.trackbodyid = -1
        self.camera2.cam.distance = self.model.stat.extent * 3.5
        self.camera2.cam.elevation = -50
        self.camera2.cam.lookat[0] += 0.1
        self.camera2.cam.lookat[1] -= 0.1
        # self.camera2.cam.lookat[2] += 0.2

    def camera3_setup(self):
        self.camera3.cam.trackbodyid = -1
        self.camera3.cam.distance = self.model.stat.extent * 3.5
        self.camera3.cam.elevation = -50
        self.camera3.cam.lookat[0] += 0.1
        self.camera3.cam.lookat[1] -= 0.1


class Box3dFixedReachPixelMulMulObjConAvoidEnvOneEasy(mujoco_env.MujocoPixel2CamEnv, utils.EzPickle):
    def __init__(self, width=84, height=84, num_step=4):
        self.num_step = num_step
        mujoco_env.MujocoPixel2CamEnv.__init__(self, 'arm_reach_one.xml', 4, width, height, "grey")
        utils.EzPickle.__init__(self)
        self.observation_space = Box(low=0, high=255, shape=(width, height, 2 * num_step))
        
    def check_contact(self):
        d = self.unwrapped.data
        for coni in range(d.ncon):
            con = d.obj.contact[coni]
            if  (con.geom1 != 0) \
            and (con.geom2 == 3):
                # Small box is touched but not by table
                return True
        return False

    def check_obj_contact(self):
        return False

    def check_table_contact(self):
        d = self.unwrapped.data
        for coni in range(d.ncon):
            con = d.obj.contact[coni]
            if  (con.geom1 == 0) \
            and (con.geom2 == 1 or con.geom2 == 2):
                # Small box is touched but not by table
                return True
        return False

    def _step(self, a):
        obses = []
        for _ in range(self.num_step):
            self.do_simulation(a, self.frame_skip)
            obses.append(self._get_obs())

        done = False
        obs = np.concatenate(obses, axis=2)
        
        reach_reward = 0.0
        contact_reward = 0.0
        table_contact_reward = 0.0

        if self.check_contact():
            contact_reward += 1.0
        if self.check_table_contact():
            table_contact_reward += 1.0

        # Check for distance
        d = self.unwrapped.data
        distance = np.linalg.norm(d.site_xpos.flatten() - list(d.qpos[4:7].flatten()))
        if distance <= 0.4:
            reach_reward += 2.0 - distance*3

        reward = reach_reward

        joint_info = np.concatenate([
           self.model.data.qpos.flat[:4],
           self.model.data.qvel.flat[:4]
        ])

        return obs, reward, done, dict(
            contact_reward=contact_reward, 
            table_contact_reward=table_contact_reward,
            joint_info=joint_info)


    def reset_model(self):
        c = 0.2
        # qpos = [0.000, 3.133, 0.018, -1.500, -0.004, -0.000, 0.005, -0.001, 0.007]
        while True:
            qpos = self.init_qpos.copy()

            qpos[:4] = [2.94179085e-02, 3.17722328e+00, -2.95601665e-01, -1.56731661e+00]

            theta = np.random.uniform(low=0.0, high=2 * math.pi, size=1)
            offset = 0.4 + np.random.uniform(low=-c, high=c, size=1)
            qpos[4] += offset * math.cos(theta)
            qpos[5] += offset * math.sin(theta)

            self.set_state(
                np.array(qpos),
                self.init_qvel,
            )

            if not self.check_obj_contact():
                break
                
        obs = self._get_obs()
        return np.concatenate([obs for _ in range(self.num_step)], axis=2)
    
    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = self.model.stat.extent * 3.5
        self.viewer.cam.elevation = -50
        self.viewer.cam.lookat[0] += 0.1
        self.viewer.cam.lookat[1] -= 0.1
        # self.viewer.cam.lookat[2] += 0.2



    def camera2_setup(self):
        self.camera2.cam.trackbodyid = -1
        self.camera2.cam.distance = self.model.stat.extent * 3.5
        self.camera2.cam.elevation = -50
        self.camera2.cam.lookat[0] += 0.1
        self.camera2.cam.lookat[1] -= 0.1
        # self.camera2.cam.lookat[2] += 0.2

    def camera3_setup(self):
        self.camera3.cam.trackbodyid = -1
        self.camera3.cam.distance = self.model.stat.extent * 3.5
        self.camera3.cam.elevation = -50
        self.camera3.cam.lookat[0] += 0.1
        self.camera3.cam.lookat[1] -= 0.1



class Box3dFixedReachHarderEnv2Step(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'arm_reach_one.xml', 4)
        utils.EzPickle.__init__(self)
        
    def check_contact(self):
        d = self.unwrapped.data
        for coni in range(d.ncon):
            con = d.obj.contact[coni]
            if  (con.geom1 != 0) \
            and (con.geom2 == 3):
                # Small box is touched but not by table
                return True
        return False

    def check_obj_contact(self):
        d = self.unwrapped.data
        for coni in range(d.ncon):
            con = d.obj.contact[coni]
            if  (con.geom1 == 0) \
            and (con.geom2 == 1 or con.geom2 == 2):
                # Small box is touched but not by table
                return True
        return False

    def check_table_contact(self):
        d = self.unwrapped.data
        for coni in range(d.ncon):
            con = d.obj.contact[coni]
            if  (con.geom1 == 0) \
            and (con.geom2 == 1 or con.geom2 == 2):
                # Small box is touched but not by table
                return True
        return False

    def _step(self, a):
        for _ in range(2):
            self.do_simulation(a, self.frame_skip)
        
        obs = self._get_obs()

        done = False
        
        reach_reward = 0.0
        contact_reward = 0.0
        table_contact_reward = 0.0

        # Check for distance
        d = self.unwrapped.data
        distance = np.linalg.norm(d.site_xpos.flatten() - list(d.qpos[4:7].flatten()))
        if distance <= 0.1:
            reach_reward += 2.0 - distance*3

        if self.check_contact():
            contact_reward = 1.0
        if self.check_table_contact():
            table_contact_reward = 1.0


        reward = reach_reward

        return obs, reward, done, dict(contact_reward=contact_reward, table_contact_reward=table_contact_reward)

    def reset_model(self):
        c = 0.05
        # qpos = [0.000, 3.133, 0.018, -1.500, -0.004, -0.000, 0.005, -0.001, 0.007]
        while True:
            qpos = self.init_qpos.copy()

            qpos[:4] = [2.94179085e-02, 3.17722328e+00, -2.95601665e-01, -1.56731661e+00]

            theta = np.random.uniform(low=0.0, high=2 * math.pi, size=1)
            offset = 0.4 + np.random.uniform(low=-c, high=c, size=1)
            qpos[4] += offset * math.cos(theta)
            qpos[5] += offset * math.sin(theta)
            # qpos[:5] += self.np_random.uniform(low=-0.1, high=0.1, size=5)
            # qpos[5:7] += self.np_random.uniform(low=-c, high=c, size=2)

            self.set_state(
                np.array(qpos),
                self.init_qvel,
            )

            if not self.check_obj_contact():
                break
        return self._get_obs()

    def _get_obs(self):
        # return self.model.data.qpos.flat[:9]
        return np.concatenate([
           self.model.data.qpos.flat,
           self.model.data.qvel.flat
        ])

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = self.model.stat.extent * 4.0
        self.viewer.cam.elevation = -40



class Box3dFixedReachHarderEnv4Step(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'arm_reach_one.xml', 4)
        utils.EzPickle.__init__(self)
        
    def check_contact(self):
        d = self.unwrapped.data
        for coni in range(d.ncon):
            con = d.obj.contact[coni]
            if  (con.geom1 != 0) \
            and (con.geom2 == 3):
                # Small box is touched but not by table
                return True
        return False

    def check_obj_contact(self):
        d = self.unwrapped.data
        for coni in range(d.ncon):
            con = d.obj.contact[coni]
            if  (con.geom1 == 0) \
            and (con.geom2 == 1 or con.geom2 == 2):
                # Small box is touched but not by table
                return True
        return False

    def check_table_contact(self):
        d = self.unwrapped.data
        for coni in range(d.ncon):
            con = d.obj.contact[coni]
            if  (con.geom1 == 0) \
            and (con.geom2 == 1 or con.geom2 == 2):
                # Small box is touched but not by table
                return True
        return False

    def _step(self, a):
        for _ in range(4):
            self.do_simulation(a, self.frame_skip)
        
        obs = self._get_obs()

        done = False
        
        reach_reward = 0.0
        contact_reward = 0.0
        table_contact_reward = 0.0

        # Check for distance
        d = self.unwrapped.data
        distance = np.linalg.norm(d.site_xpos.flatten() - list(d.qpos[4:7].flatten()))
        if distance <= 0.1:
            reach_reward += 2.0 - distance*3

        if self.check_contact():
            contact_reward = 1.0
        if self.check_table_contact():
            table_contact_reward = 1.0


        reward = reach_reward

        return obs, reward, done, dict(contact_reward=contact_reward, table_contact_reward=table_contact_reward)

    def reset_model(self):
        c = 0.05
        # qpos = [0.000, 3.133, 0.018, -1.500, -0.004, -0.000, 0.005, -0.001, 0.007]
        while True:
            qpos = self.init_qpos.copy()

            qpos[:4] = [2.94179085e-02, 3.17722328e+00, -2.95601665e-01, -1.56731661e+00]

            theta = np.random.uniform(low=0.0, high=2 * math.pi, size=1)
            offset = 0.4 + np.random.uniform(low=-c, high=c, size=1)
            qpos[4] += offset * math.cos(theta)
            qpos[5] += offset * math.sin(theta)
            # qpos[:5] += self.np_random.uniform(low=-0.1, high=0.1, size=5)
            # qpos[5:7] += self.np_random.uniform(low=-c, high=c, size=2)

            self.set_state(
                np.array(qpos),
                self.init_qvel,
            )

            if not self.check_obj_contact():
                break
        return self._get_obs()

    def _get_obs(self):
        # return self.model.data.qpos.flat[:9]
        return np.concatenate([
           self.model.data.qpos.flat,
           self.model.data.qvel.flat
        ])

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = self.model.stat.extent * 4.0
        self.viewer.cam.elevation = -40


class Box3dFixedReachHarderEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'arm_reach_one.xml', 4)
        utils.EzPickle.__init__(self)
        
    def check_contact(self):
        d = self.unwrapped.data
        for coni in range(d.ncon):
            con = d.obj.contact[coni]
            if  (con.geom1 != 0) \
            and (con.geom2 == 3):
                # Small box is touched but not by table
                return True
        return False

    def check_obj_contact(self):
        d = self.unwrapped.data
        for coni in range(d.ncon):
            con = d.obj.contact[coni]
            if  (con.geom1 == 0) \
            and (con.geom2 == 1 or con.geom2 == 2):
                # Small box is touched but not by table
                return True
        return False

    def check_table_contact(self):
        d = self.unwrapped.data
        for coni in range(d.ncon):
            con = d.obj.contact[coni]
            if  (con.geom1 == 0) \
            and (con.geom2 == 1 or con.geom2 == 2):
                # Small box is touched but not by table
                return True
        return False

    def _step(self, a):
        self.do_simulation(a, self.frame_skip)
        
        obs = self._get_obs()

        done = False
        
        reach_reward = 0.0
        contact_reward = 0.0
        table_contact_reward = 0.0

        # Check for distance
        d = self.unwrapped.data
        distance = np.linalg.norm(d.site_xpos.flatten() - list(d.qpos[4:7].flatten()))
        if distance <= 0.1:
            reach_reward += 2.0 - distance*3

        if self.check_contact():
            contact_reward = 1.0
        if self.check_table_contact():
            table_contact_reward = 1.0


        reward = reach_reward

        return obs, reward, done, dict(contact_reward=contact_reward, table_contact_reward=table_contact_reward)

    def reset_model(self):
        c = 0.05
        # qpos = [0.000, 3.133, 0.018, -1.500, -0.004, -0.000, 0.005, -0.001, 0.007]
        while True:
            qpos = self.init_qpos.copy()

            qpos[:4] = [2.94179085e-02, 3.17722328e+00, -2.95601665e-01, -1.56731661e+00]

            theta = np.random.uniform(low=0.0, high=2 * math.pi, size=1)
            offset = 0.4 + np.random.uniform(low=-c, high=c, size=1)
            qpos[4] += offset * math.cos(theta)
            qpos[5] += offset * math.sin(theta)
            # qpos[:5] += self.np_random.uniform(low=-0.1, high=0.1, size=5)
            # qpos[5:7] += self.np_random.uniform(low=-c, high=c, size=2)

            self.set_state(
                np.array(qpos),
                self.init_qvel,
            )

            if not self.check_obj_contact():
                break
        return self._get_obs()

    def _get_obs(self):
        # return self.model.data.qpos.flat[:9]
        return np.concatenate([
           self.model.data.qpos.flat,
           self.model.data.qvel.flat
        ])

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = self.model.stat.extent * 4.0
        self.viewer.cam.elevation = -40



class Box3dFixedReachPixelMulMulObjConAvoidEnvOne(mujoco_env.MujocoPixel2CamEnv, utils.EzPickle):
    def __init__(self, width=84, height=84, num_step=2):
        self.num_step = num_step
        mujoco_env.MujocoPixel2CamEnv.__init__(self, 'arm_reach_one.xml', 4, width, height, "grey")
        utils.EzPickle.__init__(self)
        self.observation_space = Box(low=0, high=255, shape=(width, height, 2 * num_step))
        
    def check_contact(self):
        d = self.unwrapped.data
        for coni in range(d.ncon):
            con = d.obj.contact[coni]
            if  (con.geom1 != 0) \
            and (con.geom2 == 3):
                # Small box is touched but not by table
                return True
        return False

    def check_obj_contact(self):
        return False

    def check_table_contact(self):
        d = self.unwrapped.data
        for coni in range(d.ncon):
            con = d.obj.contact[coni]
            if  (con.geom1 == 0) \
            and (con.geom2 == 1 or con.geom2 == 2):
                # Small box is touched but not by table
                return True
        return False

    def _step(self, a):
        obses = []
        for _ in range(self.num_step):
            self.do_simulation(a, self.frame_skip)
            obses.append(self._get_obs())

        done = False
        obs = np.concatenate(obses, axis=2)
        
        contact_reward = 0.0
        table_contact_reward = 0.0

        if self.check_contact():
            contact_reward += 1.0
        if self.check_table_contact():
            table_contact_reward += 1.0

        reward = contact_reward

        return obs, reward, done, dict(contact_reward=contact_reward, table_contact_reward=table_contact_reward)


    def reset_model(self):
        c = 0.5
        # qpos = [0.000, 3.133, 0.018, -1.500, -0.004, -0.000, 0.005, -0.001, 0.007]
        while True:
            qpos = self.init_qpos.copy()

            qpos[:4] = [2.94179085e-02, 3.17722328e+00, -2.95601665e-01, -1.56731661e+00]

            theta = np.random.uniform(low=0.0, high=2 * math.pi, size=1)
            offset = 0.4 + np.random.uniform(low=-c, high=c, size=1)
            qpos[4] += offset * math.cos(theta)
            qpos[5] += offset * math.sin(theta)

            self.set_state(
                np.array(qpos),
                self.init_qvel,
            )

            if not self.check_obj_contact():
                break
                
        obs = self._get_obs()
        return np.concatenate([obs for _ in range(self.num_step)], axis=2)
    
    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = self.model.stat.extent * 3.5
        self.viewer.cam.elevation = -50
        self.viewer.cam.lookat[0] += 0.1
        self.viewer.cam.lookat[1] -= 0.1
        # self.viewer.cam.lookat[2] += 0.2



    def camera2_setup(self):
        self.camera2.cam.trackbodyid = -1
        self.camera2.cam.distance = self.model.stat.extent * 3.5
        self.camera2.cam.elevation = -50
        self.camera2.cam.lookat[0] += 0.1
        self.camera2.cam.lookat[1] -= 0.1
        # self.camera2.cam.lookat[2] += 0.2

    def camera3_setup(self):
        self.camera3.cam.trackbodyid = -1
        self.camera3.cam.distance = self.model.stat.extent * 3.5
        self.camera3.cam.elevation = -50
        self.camera3.cam.lookat[0] += 0.1
        self.camera3.cam.lookat[1] -= 0.1
        # self.camera3.cam.lookat[2] += 0.2


class Box3dFixedReachMulMulObjConAvoidEnvOne(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'arm_reach_one.xml', 4)
        utils.EzPickle.__init__(self)
        
    def check_contact(self):
        d = self.unwrapped.data
        for coni in range(d.ncon):
            con = d.obj.contact[coni]
            if  (con.geom1 != 0) \
            and (con.geom2 == 3):
                # Small box is touched but not by table
                return True
        return False

    def check_obj_contact(self):
        return False

    def check_table_contact(self):
        d = self.unwrapped.data
        for coni in range(d.ncon):
            con = d.obj.contact[coni]
            if  (con.geom1 == 0) \
            and (con.geom2 == 1 or con.geom2 == 2):
                # Small box is touched but not by table
                return True
        return False

    def _step(self, a):
        self.do_simulation(a, self.frame_skip)
        
        obs = self._get_obs()

        done = False
        
        contact_reward = 0.0
        table_contact_reward = 0.0

        # Check for distance
        if self.check_contact():
            contact_reward = 1.0
        if self.check_table_contact():
            table_contact_reward = 1.0


        reward = contact_reward

        return obs, reward, done, dict(contact_reward=contact_reward, table_contact_reward=table_contact_reward)

    def reset_model(self):
        c = 0.05
        # qpos = [0.000, 3.133, 0.018, -1.500, -0.004, -0.000, 0.005, -0.001, 0.007]
        while True:
            qpos = self.init_qpos.copy()

            qpos[:4] = [2.94179085e-02, 3.17722328e+00, -2.95601665e-01, -1.56731661e+00]

            theta = np.random.uniform(low=0.0, high=math.pi, size=1)
            offset = 0.4 + np.random.uniform(low=-c, high=c, size=1)
            qpos[4] += offset * math.cos(theta)
            qpos[5] += offset * math.sin(theta)
            # qpos[:5] += self.np_random.uniform(low=-0.1, high=0.1, size=5)
            # qpos[5:7] += self.np_random.uniform(low=-c, high=c, size=2)

            self.set_state(
                np.array(qpos),
                self.init_qvel,
            )

            if not self.check_obj_contact():
                break
        return self._get_obs()

    def _get_obs(self):
        # return self.model.data.qpos.flat[:9]
        return np.concatenate([
           self.model.data.qpos.flat,
           self.model.data.qvel.flat
        ])

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = self.model.stat.extent * 4.0
        self.viewer.cam.elevation = -40


class Box3dFixedReachEnvPixelGreyMulMulContactTwoCamMulActNoMas(mujoco_env.MujocoPixel2CamEnv, utils.EzPickle):
    def __init__(self, width=84, height=84, num_step=4):
        self.num_step = num_step
        mujoco_env.MujocoPixel2CamEnv.__init__(self, 'arm_reach_6_boxes_big_no_mas.xml', 4, width, height, "grey")
        utils.EzPickle.__init__(self)
        self.observation_space = Box(low=0, high=255, shape=(width, height, 2 * num_step))

    def check_contact(self):
        d = self.unwrapped.data
        for coni in range(d.ncon):
            con = d.obj.contact[coni]
            if  (con.geom1 != 0 and \
                 con.geom1 != 3 and \
                 con.geom1 != 4 and \
                 con.geom1 != 5 and \
                 con.geom1 != 6 and \
                 con.geom1 != 7 and \
                 con.geom1 != 8) \
            and (con.geom2 == 3 or \
                 con.geom2 == 4 or \
                 con.geom2 == 5 or \
                 con.geom2 == 6 or \
                 con.geom2 == 7 or \
                 con.geom2 == 8):
                # Small box is touched but not by table
                return True
        return False

    def check_obj_contact(self):
        d = self.unwrapped.data
        for coni in range(d.ncon):
            con = d.obj.contact[coni]
            if  (con.geom1 == 3 or \
                 con.geom1 == 4 or \
                 con.geom1 == 5 or \
                 con.geom1 == 6 or \
                 con.geom1 == 7 or \
                 con.geom1 == 8) \
            and (con.geom2 == 3 or \
                 con.geom2 == 4 or \
                 con.geom2 == 5 or \
                 con.geom2 == 6 or \
                 con.geom2 == 7 or \
                 con.geom2 == 8):
                # Small box is touched but not by table
                return True
        return False

    def _step(self, a):
        obses = []
        for _ in range(self.num_step):
            self.do_simulation(a, self.frame_skip)
            obses.append(self._get_obs())

        done = False
        obs = np.concatenate(obses, axis=2)
        
        reach_reward = 0.0
        contact_reward = 0.0

        if self.check_contact():
            contact_reward += 1.0

        reward = contact_reward + reach_reward

        return obs, reward, done, dict(reach_reward=reach_reward, contact_reward=contact_reward)


    def reset_model(self):
        c = 0.5
        # qpos = [0.000, 3.133, 0.018, -1.500, -0.004, -0.000, 0.005, -0.001, 0.007]
        while True:
            qpos = self.init_qpos.copy()
            qpos[:4] = [0.000, 1.000, 0.000, -0.004] + self.np_random.uniform(low=-0.1, high=0.1, size=4)
            qpos[4:6] += self.np_random.uniform(low=-c, high=c, size=2)
            qpos[11:13] += self.np_random.uniform(low=-c, high=c, size=2)
            qpos[18:20] += self.np_random.uniform(low=-c, high=c, size=2)
            qpos[25:27] += self.np_random.uniform(low=-c, high=c, size=2)
            qpos[32:34] += self.np_random.uniform(low=-c, high=c, size=2)
            qpos[39:41] += self.np_random.uniform(low=-c, high=c, size=2)


            self.set_state(
                np.array(qpos),
                self.init_qvel,
            )

            if not self.check_obj_contact():
                break
                
        obs = self._get_obs()
        return np.concatenate([obs for _ in range(self.num_step)], axis=2)
    
    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = self.model.stat.extent * 2.5
        self.viewer.cam.elevation = -40
        self.viewer.cam.lookat[0] += 0.2
        self.viewer.cam.lookat[1] -= 0.1


    def camera2_setup(self):
        self.camera2.cam.trackbodyid = -1
        self.camera2.cam.distance = self.model.stat.extent * 2.5
        self.camera2.cam.elevation = -40
        self.camera2.cam.lookat[0] += 0.2
        self.camera2.cam.lookat[1] -= 0.1

    def camera3_setup(self):
        self.camera3.cam.trackbodyid = -1
        self.camera3.cam.distance = self.model.stat.extent * 2.5
        self.camera3.cam.elevation = -40
        self.camera3.cam.lookat[0] += 0.2
        self.camera3.cam.lookat[1] -= 0.1




class Box3dFixedReachEnvPixelGreyMulMulContactTwoCamMulActFusion(mujoco_env.MujocoPixel2CamEnv, utils.EzPickle):
    def __init__(self, width=84, height=84, num_step=4):
        self.num_step = num_step
        mujoco_env.MujocoPixel2CamEnv.__init__(self, 'arm_reach_mul_mul_limit.xml', 4, width, height, "grey")
        utils.EzPickle.__init__(self)
        self.observation_space = Box(low=0, high=255, shape=(width, height, 2 * num_step))

    def check_contact(self):
        d = self.unwrapped.data
        for coni in range(d.ncon):
            con = d.obj.contact[coni]
            if  (con.geom1 != 0 and \
                 con.geom1 != 3 and \
                 con.geom1 != 4 and \
                 con.geom1 != 5 and \
                 con.geom1 != 6 and \
                 con.geom1 != 7 and \
                 con.geom1 != 8) \
            and (con.geom2 == 3 or \
                 con.geom2 == 4 or \
                 con.geom2 == 5 or \
                 con.geom2 == 6 or \
                 con.geom2 == 7 or \
                 con.geom2 == 8):
                # Small box is touched but not by table
                return True
        return False

    def check_obj_contact(self):
        d = self.unwrapped.data
        for coni in range(d.ncon):
            con = d.obj.contact[coni]
            if  (con.geom1 == 3 or \
                 con.geom1 == 4 or \
                 con.geom1 == 5 or \
                 con.geom1 == 6 or \
                 con.geom1 == 7 or \
                 con.geom1 == 8) \
            and (con.geom2 == 3 or \
                 con.geom2 == 4 or \
                 con.geom2 == 5 or \
                 con.geom2 == 6 or \
                 con.geom2 == 7 or \
                 con.geom2 == 8):
                # Small box is touched but not by table
                return True
        return False


    def _step(self, a):
        obses = []
        for _ in range(self.num_step):
            self.do_simulation(a, self.frame_skip)
            obses.append(self._get_obs())

        done = False
        obs = np.concatenate(obses, axis=2)
        
        reach_reward = 0.0
        contact_reward = 0.0

        if self.check_contact():
            contact_reward += 1.0

        reward = contact_reward + reach_reward
        joint_info = np.concatenate([self.model.data.qpos.flat[:4], self.model.data.qvel.flat[:4]])

        return obs, reward, done, dict(reach_reward=reach_reward, contact_reward=contact_reward, joint_info=joint_info)


    def reset_model(self):
        c = 0.2
        # qpos = [0.000, 3.133, 0.018, -1.500, -0.004, -0.000, 0.005, -0.001, 0.007]
        while True:
            qpos = self.init_qpos.copy()
            qpos[:4] = [0.000, 1.000, -1.000, -0.004] + self.np_random.uniform(low=-0.1, high=0.1, size=4)
            qpos[4:6] += self.np_random.uniform(low=-c, high=c, size=2)
            qpos[11:13] += self.np_random.uniform(low=-c, high=c, size=2)
            qpos[18:20] += self.np_random.uniform(low=-c, high=c, size=2)
            qpos[25:27] += self.np_random.uniform(low=-c, high=c, size=2)
            qpos[32:34] += self.np_random.uniform(low=-c, high=c, size=2)
            qpos[39:41] += self.np_random.uniform(low=-c, high=c, size=2)


            self.set_state(
                np.array(qpos),
                self.init_qvel,
            )

            if not self.check_obj_contact():
                break
                
        obs = self._get_obs()
        joint_info = np.concatenate([self.model.data.qpos.flat[:4], self.model.data.qvel.flat[:4]])
        return np.concatenate([obs for _ in range(self.num_step)], axis=2), dict(joint_info=joint_info)
    
    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = self.model.stat.extent * 2.5
        self.viewer.cam.elevation = -40

    def camera2_setup(self):
        self.camera2.cam.trackbodyid = -1
        self.camera2.cam.distance = self.model.stat.extent * 2.5*1.4
        self.camera2.cam.elevation = -40

    def camera3_setup(self):
        self.camera3.cam.trackbodyid = -1
        self.camera3.cam.distance = self.model.stat.extent * 2.5*1.4
        self.camera3.cam.elevation = -40



class Box3dFixedReachEnvPixelGreyMulMulContactTwoCamMulAct(mujoco_env.MujocoPixel2CamEnv, utils.EzPickle):
    def __init__(self, width=84, height=84, num_step=4):
        self.num_step = num_step
        mujoco_env.MujocoPixel2CamEnv.__init__(self, 'arm_reach_6_boxes_big.xml', 4, width, height, "grey")
        utils.EzPickle.__init__(self)
        self.observation_space = Box(low=0, high=255, shape=(width, height, 2 * num_step))

    def check_contact(self):
        d = self.unwrapped.data
        for coni in range(d.ncon):
            con = d.obj.contact[coni]
            if  (con.geom1 != 0 and \
                 con.geom1 != 3 and \
                 con.geom1 != 4 and \
                 con.geom1 != 5 and \
                 con.geom1 != 6 and \
                 con.geom1 != 7 and \
                 con.geom1 != 8) \
            and (con.geom2 == 3 or \
                 con.geom2 == 4 or \
                 con.geom2 == 5 or \
                 con.geom2 == 6 or \
                 con.geom2 == 7 or \
                 con.geom2 == 8):
                # Small box is touched but not by table
                return True
        return False

    def check_obj_contact(self):
        d = self.unwrapped.data
        for coni in range(d.ncon):
            con = d.obj.contact[coni]
            if  (con.geom1 == 3 or \
                 con.geom1 == 4 or \
                 con.geom1 == 5 or \
                 con.geom1 == 6 or \
                 con.geom1 == 7 or \
                 con.geom1 == 8) \
            and (con.geom2 == 3 or \
                 con.geom2 == 4 or \
                 con.geom2 == 5 or \
                 con.geom2 == 6 or \
                 con.geom2 == 7 or \
                 con.geom2 == 8):
                # Small box is touched but not by table
                return True
        return False

    def check_table_contact(self):
        d = self.unwrapped.data
        for coni in range(d.ncon):
            con = d.obj.contact[coni]
            if  (con.geom1 == 0) \
            and (con.geom2 == 1 or con.geom2 == 2):
                # Small box is touched but not by table
                return True
        return False

    def _step(self, a):
        obses = []
        for _ in range(self.num_step):
            self.do_simulation(a, self.frame_skip)
            obses.append(self._get_obs())

        done = False
        obs = np.concatenate(obses, axis=2)
        
        contact_reward = 0.0
        table_contact_reward = 0.0

        if self.check_contact():
            contact_reward += 1.0
        if self.check_table_contact():
            table_contact_reward += 1.0

        reward = contact_reward

        return obs, reward, done, dict(contact_reward=contact_reward, table_contact_reward=table_contact_reward)


    def reset_model(self):
        c = 0.5
        # qpos = [0.000, 3.133, 0.018, -1.500, -0.004, -0.000, 0.005, -0.001, 0.007]
        while True:
            qpos = self.init_qpos.copy()
            qpos[:4] = [0.000, 1.000, 0.000, -0.004] + self.np_random.uniform(low=-0.1, high=0.1, size=4)
            qpos[4:6] += self.np_random.uniform(low=-c, high=c, size=2)
            qpos[11:13] += self.np_random.uniform(low=-c, high=c, size=2)
            qpos[18:20] += self.np_random.uniform(low=-c, high=c, size=2)
            qpos[25:27] += self.np_random.uniform(low=-c, high=c, size=2)
            qpos[32:34] += self.np_random.uniform(low=-c, high=c, size=2)
            qpos[39:41] += self.np_random.uniform(low=-c, high=c, size=2)


            self.set_state(
                np.array(qpos),
                self.init_qvel,
            )

            if not self.check_obj_contact():
                break
                
        obs = self._get_obs()
        return np.concatenate([obs for _ in range(self.num_step)], axis=2)
    
    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = self.model.stat.extent * 2.5
        self.viewer.cam.elevation = -40
        self.viewer.cam.lookat[0] += 0.2
        self.viewer.cam.lookat[1] -= 0.1


    def camera2_setup(self):
        self.camera2.cam.trackbodyid = -1
        self.camera2.cam.distance = self.model.stat.extent * 2.5
        self.camera2.cam.elevation = -40
        self.camera2.cam.lookat[0] += 0.2
        self.camera2.cam.lookat[1] -= 0.1

    def camera3_setup(self):
        self.camera3.cam.trackbodyid = -1
        self.camera3.cam.distance = self.model.stat.extent * 2.5
        self.camera3.cam.elevation = -40
        self.camera3.cam.lookat[0] += 0.2
        self.camera3.cam.lookat[1] -= 0.1


class Box3dFixedReachEnvPixelGreyMulMulContactTwoCamAct(mujoco_env.MujocoPixel2CamEnv, utils.EzPickle):
    def __init__(self, width=84, height=84, num_step=2):
        self.num_step = num_step
        mujoco_env.MujocoPixel2CamEnv.__init__(self, 'arm_reach_6_boxes_big.xml', 4, width, height, "grey")
        utils.EzPickle.__init__(self)
        self.observation_space = Box(low=0, high=255, shape=(width, height, 2 * num_step))

    def check_contact(self):
        d = self.unwrapped.data
        for coni in range(d.ncon):
            con = d.obj.contact[coni]
            if  (con.geom1 != 0 and \
                 con.geom1 != 3 and \
                 con.geom1 != 4 and \
                 con.geom1 != 5 and \
                 con.geom1 != 6 and \
                 con.geom1 != 7 and \
                 con.geom1 != 8) \
            and (con.geom2 == 3 or \
                 con.geom2 == 4 or \
                 con.geom2 == 5 or \
                 con.geom2 == 6 or \
                 con.geom2 == 7 or \
                 con.geom2 == 8):
                # Small box is touched but not by table
                return True
        return False

    def check_obj_contact(self):
        d = self.unwrapped.data
        for coni in range(d.ncon):
            con = d.obj.contact[coni]
            if  (con.geom1 == 3 or \
                 con.geom1 == 4 or \
                 con.geom1 == 5 or \
                 con.geom1 == 6 or \
                 con.geom1 == 7 or \
                 con.geom1 == 8) \
            and (con.geom2 == 3 or \
                 con.geom2 == 4 or \
                 con.geom2 == 5 or \
                 con.geom2 == 6 or \
                 con.geom2 == 7 or \
                 con.geom2 == 8):
                # Small box is touched but not by table
                return True
        return False

    def check_table_contact(self):
        d = self.unwrapped.data
        for coni in range(d.ncon):
            con = d.obj.contact[coni]
            if  (con.geom1 == 0) \
            and (con.geom2 == 1 or con.geom2 == 2):
                # Small box is touched but not by table
                return True
        return False

    def _step(self, a):
        obses = []
        for _ in range(self.num_step):
            self.do_simulation(a, self.frame_skip)
            obses.append(self._get_obs())

        done = False
        obs = np.concatenate(obses, axis=2)
        
        contact_reward = 0.0
        table_contact_reward = 0.0

        if self.check_contact():
            contact_reward += 1.0
        if self.check_table_contact():
            table_contact_reward += 1.0

        reward = contact_reward

        return obs, reward, done, dict(contact_reward=contact_reward, table_contact_reward=table_contact_reward)


    def reset_model(self):
        c = 0.5
        # qpos = [0.000, 3.133, 0.018, -1.500, -0.004, -0.000, 0.005, -0.001, 0.007]
        while True:
            qpos = self.init_qpos.copy()
            qpos[:4] = [0.000, 1.000, 0.000, -0.004] + self.np_random.uniform(low=-0.1, high=0.1, size=4)
            qpos[4:6] += self.np_random.uniform(low=-c, high=c, size=2)
            qpos[11:13] += self.np_random.uniform(low=-c, high=c, size=2)
            qpos[18:20] += self.np_random.uniform(low=-c, high=c, size=2)
            qpos[25:27] += self.np_random.uniform(low=-c, high=c, size=2)
            qpos[32:34] += self.np_random.uniform(low=-c, high=c, size=2)
            qpos[39:41] += self.np_random.uniform(low=-c, high=c, size=2)


            self.set_state(
                np.array(qpos),
                self.init_qvel,
            )

            if not self.check_obj_contact():
                break
                
        obs = self._get_obs()
        return np.concatenate([obs for _ in range(self.num_step)], axis=2)
    
    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = self.model.stat.extent * 2.5
        self.viewer.cam.elevation = -40
        self.viewer.cam.lookat[0] += 0.2
        self.viewer.cam.lookat[1] -= 0.1


    def camera2_setup(self):
        self.camera2.cam.trackbodyid = -1
        self.camera2.cam.distance = self.model.stat.extent * 2.5
        self.camera2.cam.elevation = -40
        self.camera2.cam.lookat[0] += 0.2
        self.camera2.cam.lookat[1] -= 0.1

    def camera3_setup(self):
        self.camera3.cam.trackbodyid = -1
        self.camera3.cam.distance = self.model.stat.extent * 2.5
        self.camera3.cam.elevation = -40
        self.camera3.cam.lookat[0] += 0.2
        self.camera3.cam.lookat[1] -= 0.1

class Box3dFixedReachEnvPixelGreyMulMulContactTwoCamMulActLess(mujoco_env.MujocoPixel2CamEnv, utils.EzPickle):
    def __init__(self, width=84, height=84, num_step=2):
        self.num_step = num_step
        mujoco_env.MujocoPixel2CamEnv.__init__(self, 'arm_reach_mul_mul_limit.xml', 4, width, height, "grey")
        utils.EzPickle.__init__(self)
        self.observation_space = Box(low=0, high=255, shape=(width, height, 2 * num_step))

    def check_contact(self):
        d = self.unwrapped.data
        for coni in range(d.ncon):
            con = d.obj.contact[coni]
            if  (con.geom1 != 0 and \
                 con.geom1 != 3 and \
                 con.geom1 != 4 and \
                 con.geom1 != 5 and \
                 con.geom1 != 6 and \
                 con.geom1 != 7 and \
                 con.geom1 != 8) \
            and (con.geom2 == 3 or \
                 con.geom2 == 4 or \
                 con.geom2 == 5 or \
                 con.geom2 == 6 or \
                 con.geom2 == 7 or \
                 con.geom2 == 8):
                # Small box is touched but not by table
                return True
        return False

    def check_obj_contact(self):
        d = self.unwrapped.data
        for coni in range(d.ncon):
            con = d.obj.contact[coni]
            if  (con.geom1 == 3 or \
                 con.geom1 == 4 or \
                 con.geom1 == 5 or \
                 con.geom1 == 6 or \
                 con.geom1 == 7 or \
                 con.geom1 == 8) \
            and (con.geom2 == 3 or \
                 con.geom2 == 4 or \
                 con.geom2 == 5 or \
                 con.geom2 == 6 or \
                 con.geom2 == 7 or \
                 con.geom2 == 8):
                # Small box is touched but not by table
                return True
        return False

    def _step(self, a):
        obses = []
        for _ in range(self.num_step):
            self.do_simulation(a, self.frame_skip)
            obses.append(self._get_obs())

        done = False
        obs = np.concatenate(obses, axis=2)
        
        reach_reward = 0.0
        contact_reward = 0.0

        if self.check_contact():
            contact_reward += 1.0

        reward = contact_reward + reach_reward

        return obs, reward, done, dict(reach_reward=reach_reward, contact_reward=contact_reward)


    def reset_model(self):
        c = 0.2
        # qpos = [0.000, 3.133, 0.018, -1.500, -0.004, -0.000, 0.005, -0.001, 0.007]
        while True:
            qpos = self.init_qpos.copy()
            qpos[:4] = [0.000, 1.000, -1.000, -0.004] + self.np_random.uniform(low=-0.1, high=0.1, size=4)
            qpos[4:6] += self.np_random.uniform(low=-c, high=c, size=2)
            qpos[11:13] += self.np_random.uniform(low=-c, high=c, size=2)
            qpos[18:20] += self.np_random.uniform(low=-c, high=c, size=2)
            qpos[25:27] += self.np_random.uniform(low=-c, high=c, size=2)
            qpos[32:34] += self.np_random.uniform(low=-c, high=c, size=2)
            qpos[39:41] += self.np_random.uniform(low=-c, high=c, size=2)


            self.set_state(
                np.array(qpos),
                self.init_qvel,
            )

            if not self.check_obj_contact():
                break
                
        obs = self._get_obs()
        return np.concatenate([obs for _ in range(self.num_step)], axis=2)
    
    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = self.model.stat.extent * 2.5
        self.viewer.cam.elevation = -40

    def camera2_setup(self):
        self.camera2.cam.trackbodyid = -1
        self.camera2.cam.distance = self.model.stat.extent * 2.5*1.4
        self.camera2.cam.elevation = -40

    def camera3_setup(self):
        self.camera3.cam.trackbodyid = -1
        self.camera3.cam.distance = self.model.stat.extent * 2.5*1.4
        self.camera3.cam.elevation = -40


class Box3dFixedReachEnvPixelGreyMulMulTwoCamMulAct(mujoco_env.MujocoPixel2CamEnv, utils.EzPickle):
    def __init__(self, width=84, height=84, num_step=4):
        self.num_step = num_step
        mujoco_env.MujocoPixel2CamEnv.__init__(self, 'arm_reach_mul_mul_limit.xml', 4, width, height, "grey")
        utils.EzPickle.__init__(self)
        self.observation_space = Box(low=0, high=255, shape=(width, height, 2 * num_step))

    def check_contact(self):
        d = self.unwrapped.data
        for coni in range(d.ncon):
            con = d.obj.contact[coni]
            if  (con.geom1 != 0 and \
                 con.geom1 != 3 and \
                 con.geom1 != 4 and \
                 con.geom1 != 5 and \
                 con.geom1 != 6 and \
                 con.geom1 != 7 and \
                 con.geom1 != 8) \
            and (con.geom2 == 3 or \
                 con.geom2 == 4 or \
                 con.geom2 == 5 or \
                 con.geom2 == 6 or \
                 con.geom2 == 7 or \
                 con.geom2 == 8):
                # Small box is touched but not by table
                return True
        return False

    def check_obj_contact(self):
        d = self.unwrapped.data
        for coni in range(d.ncon):
            con = d.obj.contact[coni]
            if  (con.geom1 == 3 or \
                 con.geom1 == 4 or \
                 con.geom1 == 5 or \
                 con.geom1 == 6 or \
                 con.geom1 == 7 or \
                 con.geom1 == 8) \
            and (con.geom2 == 3 or \
                 con.geom2 == 4 or \
                 con.geom2 == 5 or \
                 con.geom2 == 6 or \
                 con.geom2 == 7 or \
                 con.geom2 == 8):
                # Small box is touched but not by table
                return True
        return False

    def _step(self, a):
        obses = []
        for _ in range(self.num_step):
            self.do_simulation(a, self.frame_skip)
            obses.append(self._get_obs())

        done = False
        obs = np.concatenate(obses, axis=2)
        
        reach_reward = 0.0
        contact_reward = 0.0

        if self.check_contact():
            contact_reward += 1.0

        reward = 0.0

        return obs, reward, done, dict(reach_reward=reach_reward, contact_reward=contact_reward)


    def reset_model(self):
        c = 0.2
        # qpos = [0.000, 3.133, 0.018, -1.500, -0.004, -0.000, 0.005, -0.001, 0.007]
        while True:
            qpos = self.init_qpos.copy()
            qpos[:4] = [0.000, 1.000, -1.000, -0.004] + self.np_random.uniform(low=-0.1, high=0.1, size=4)
            qpos[4:6] += self.np_random.uniform(low=-c, high=c, size=2)
            qpos[11:13] += self.np_random.uniform(low=-c, high=c, size=2)
            qpos[18:20] += self.np_random.uniform(low=-c, high=c, size=2)
            qpos[25:27] += self.np_random.uniform(low=-c, high=c, size=2)
            qpos[32:34] += self.np_random.uniform(low=-c, high=c, size=2)
            qpos[39:41] += self.np_random.uniform(low=-c, high=c, size=2)

            self.set_state(
                np.array(qpos),
                self.init_qvel,
            )

            if not self.check_obj_contact():
                break
                
        obs = self._get_obs()
        return np.concatenate([obs for _ in range(self.num_step)], axis=2)
    
    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = self.model.stat.extent * 2.5
        self.viewer.cam.elevation = -40
        self.viewer.cam.lookat[0] += 0.8
        # self.viewer.cam.lookat[1] += 0.3
        # self.viewer.cam.lookat[2] += 0


    def camera2_setup(self):
        self.camera2.cam.trackbodyid = -1
        self.camera2.cam.distance = self.model.stat.extent * 2.5*1.4
        self.camera2.cam.elevation = -40
        self.camera2.cam.lookat[0] += 0.4
        self.camera2.cam.lookat[1] += 0.3
        self.camera2.cam.lookat[2] += 0

    def camera3_setup(self):
        self.camera3.cam.trackbodyid = -1
        self.camera3.cam.distance = self.model.stat.extent * 2.5*1.4
        self.camera3.cam.elevation = -40
        self.camera3.cam.lookat[0] += 0.4
        self.camera3.cam.lookat[1] += 0.3
        self.camera3.cam.lookat[2] += 0


class Box3dFixedReachEnvPixelGreyMulMulTwoCamMulActLess(mujoco_env.MujocoPixel2CamEnv, utils.EzPickle):
    def __init__(self, width=84, height=84, num_step=2):
        self.num_step = num_step
        mujoco_env.MujocoPixel2CamEnv.__init__(self, 'arm_reach_mul_mul_limit.xml', 4, width, height, "grey")
        utils.EzPickle.__init__(self)
        self.observation_space = Box(low=0, high=255, shape=(width, height, 2 * num_step))

    def check_contact(self):
        d = self.unwrapped.data
        for coni in range(d.ncon):
            con = d.obj.contact[coni]
            if  (con.geom1 != 0 and \
                 con.geom1 != 3 and \
                 con.geom1 != 4 and \
                 con.geom1 != 5 and \
                 con.geom1 != 6 and \
                 con.geom1 != 7 and \
                 con.geom1 != 8) \
            and (con.geom2 == 3 or \
                 con.geom2 == 4 or \
                 con.geom2 == 5 or \
                 con.geom2 == 6 or \
                 con.geom2 == 7 or \
                 con.geom2 == 8):
                # Small box is touched but not by table
                return True
        return False

    def check_obj_contact(self):
        d = self.unwrapped.data
        for coni in range(d.ncon):
            con = d.obj.contact[coni]
            if  (con.geom1 == 3 or \
                 con.geom1 == 4 or \
                 con.geom1 == 5 or \
                 con.geom1 == 6 or \
                 con.geom1 == 7 or \
                 con.geom1 == 8) \
            and (con.geom2 == 3 or \
                 con.geom2 == 4 or \
                 con.geom2 == 5 or \
                 con.geom2 == 6 or \
                 con.geom2 == 7 or \
                 con.geom2 == 8):
                # Small box is touched but not by table
                return True
        return False

    def _step(self, a):
        obses = []
        for _ in range(self.num_step):
            self.do_simulation(a, self.frame_skip)
            obses.append(self._get_obs())

        done = False
        obs = np.concatenate(obses, axis=2)
        
        reach_reward = 0.0
        contact_reward = 0.0

        if self.check_contact():
            contact_reward += 1.0

        reward = 0.0

        return obs, reward, done, dict(reach_reward=reach_reward, contact_reward=contact_reward)


    def reset_model(self):
        c = 0.2
        # qpos = [0.000, 3.133, 0.018, -1.500, -0.004, -0.000, 0.005, -0.001, 0.007]
        while True:
            qpos = self.init_qpos.copy()
            qpos[:4] = [0.000, 1.000, -1.000, -0.004] + self.np_random.uniform(low=-0.1, high=0.1, size=4)
            qpos[4:6] += self.np_random.uniform(low=-c, high=c, size=2)
            qpos[11:13] += self.np_random.uniform(low=-c, high=c, size=2)
            qpos[18:20] += self.np_random.uniform(low=-c, high=c, size=2)
            qpos[25:27] += self.np_random.uniform(low=-c, high=c, size=2)
            qpos[32:34] += self.np_random.uniform(low=-c, high=c, size=2)
            qpos[39:41] += self.np_random.uniform(low=-c, high=c, size=2)

            self.set_state(
                np.array(qpos),
                self.init_qvel,
            )

            if not self.check_obj_contact():
                break

        obs = self._get_obs()
        return np.concatenate([obs for _ in range(self.num_step)], axis=2)
    
    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = self.model.stat.extent * 2.5
        self.viewer.cam.elevation = -40

    def camera2_setup(self):
        self.camera2.cam.trackbodyid = -1
        self.camera2.cam.distance = self.model.stat.extent * 2.5*1.4
        self.camera2.cam.elevation = -40

    def camera3_setup(self):
        self.camera3.cam.trackbodyid = -1
        self.camera3.cam.distance = self.model.stat.extent * 2.5*1.4
        self.camera3.cam.elevation = -40


class Box3dFixedReachEnvPixelGreyHarderTwoCamMulActLessRepeatTwo(mujoco_env.MujocoPixel2CamEnv, utils.EzPickle):
    def __init__(self, width=84, height=84, num_step=2):
        self.num_step = num_step
        mujoco_env.MujocoPixel2CamEnv.__init__(self, 'arm_claw_new_one_box.xml', 4, width, height, "grey")
        utils.EzPickle.__init__(self)
        self.observation_space = Box(low=0, high=255, shape=(width, height, 2 * num_step))


    def _step(self, a):
        obses = []
        for _ in range(self.num_step):
            self.do_simulation(a, self.frame_skip)
            self.do_simulation(a, self.frame_skip) # Repeat one action
            obses.append(self._get_obs())

        done = False
        obs = np.concatenate(obses, axis=2)
        
        reach_reward = 0.0
        contact_reward = 0.0

        # Check for distance
        d = self.unwrapped.data
        distance = np.linalg.norm(d.site_xpos.flatten() - (list(d.qpos[-2:].flatten()) + [0.075]))
        # print (distance)
        if distance <= 0.1:
            reach_reward += 2.0 - distance*3
        reward = reach_reward

        return obs, reward, done, dict(reach_reward=reach_reward, contact_reward=contact_reward)


    def reset_model(self):
        c = 0.1
        qpos = self.init_qpos.copy()
        qpos[4:6] += self.np_random.uniform(low=-c, high=c, size=2)
        # qpos = self.init_qpos
        self.set_state(
            qpos,
            self.init_qvel,
        )
        obs = self._get_obs()
        return np.concatenate([obs for _ in range(self.num_step)], axis=2)
    
    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = self.model.stat.extent * 2.5
        self.viewer.cam.elevation = -40

    def camera2_setup(self):
        self.camera2.cam.trackbodyid = -1
        self.camera2.cam.distance = self.model.stat.extent * 2.5*1.4
        self.camera2.cam.elevation = -40

    def camera3_setup(self):
        self.camera3.cam.trackbodyid = -1
        self.camera3.cam.distance = self.model.stat.extent * 2.5*1.4
        self.camera3.cam.elevation = -40



class Box3dFixedReachEnvPixelGreyHarderTwoCamMulActLess(mujoco_env.MujocoPixel2CamEnv, utils.EzPickle):
    def __init__(self, width=84, height=84, num_step=2):
        self.num_step = num_step
        mujoco_env.MujocoPixel2CamEnv.__init__(self, 'arm_claw_new_one_box.xml', 4, width, height, "grey")
        utils.EzPickle.__init__(self)
        self.observation_space = Box(low=0, high=255, shape=(width, height, 2 * num_step))


    def _step(self, a):
        obses = []
        for _ in range(self.num_step):
            self.do_simulation(a, self.frame_skip)
            obses.append(self._get_obs())

        done = False
        obs = np.concatenate(obses, axis=2)
        
        reach_reward = 0.0
        contact_reward = 0.0

        # Check for distance
        d = self.unwrapped.data
        distance = np.linalg.norm(d.site_xpos.flatten() - (list(d.qpos[-2:].flatten()) + [0.075]))
        # print (distance)
        if distance <= 0.1:
            reach_reward += 2.0 - distance*3
        reward = reach_reward

        return obs, reward, done, dict(reach_reward=reach_reward, contact_reward=contact_reward)


    def reset_model(self):
        c = 0.1
        qpos = self.init_qpos.copy()
        # qpos[:7] = [0.000, 3.133, 0.018, -1.500, -0.004, -0.000, 0.005]
        qpos[4:6] += self.np_random.uniform(low=-c, high=c, size=2)
        # qpos = self.init_qpos
        self.set_state(
            qpos,
            self.init_qvel,
        )
        obs = self._get_obs()
        return np.concatenate([obs for _ in range(self.num_step)], axis=2)
    
    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = self.model.stat.extent * 2.5
        self.viewer.cam.elevation = -40

    def camera2_setup(self):
        self.camera2.cam.trackbodyid = -1
        self.camera2.cam.distance = self.model.stat.extent * 2.5*1.4
        self.camera2.cam.elevation = -40

    def camera3_setup(self):
        self.camera3.cam.trackbodyid = -1
        self.camera3.cam.distance = self.model.stat.extent * 2.5*1.4
        self.camera3.cam.elevation = -40



class Box3dFixedReachEnvPixelGreyHarderTwoCamMulAct(mujoco_env.MujocoPixel2CamEnv, utils.EzPickle):
    def __init__(self, width=84, height=84, num_step=4):
        self.num_step = num_step
        mujoco_env.MujocoPixel2CamEnv.__init__(self, 'arm_claw_new_one_box.xml', 4, width, height, "grey")
        utils.EzPickle.__init__(self)
        self.observation_space = Box(low=0, high=255, shape=(width, height, 2 * num_step))


    def _step(self, a):
        obses = []
        for _ in range(self.num_step):
            self.do_simulation(a, self.frame_skip)
            obses.append(self._get_obs())

        done = False
        obs = np.concatenate(obses, axis=2)
        
        reach_reward = 0.0
        contact_reward = 0.0

        # Check for distance
        d = self.unwrapped.data
        distance = np.linalg.norm(d.site_xpos.flatten() - (list(d.qpos[-2:].flatten()) + [0.075]))
        # print (distance)
        if distance <= 0.1:
            reach_reward += 2.0 - distance*3
        reward = reach_reward

        return obs, reward, done, dict(reach_reward=reach_reward, contact_reward=contact_reward)


    def reset_model(self):
        c = 0.1
        qpos = self.init_qpos.copy()
        # qpos[:7] = [0.000, 3.133, 0.018, -1.500, -0.004, -0.000, 0.005]
        qpos[4:6] += self.np_random.uniform(low=-c, high=c, size=2)
        # qpos = self.init_qpos
        self.set_state(
            qpos,
            self.init_qvel,
        )
        obs = self._get_obs()
        return np.concatenate([obs for _ in range(self.num_step)], axis=2)
    
    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = self.model.stat.extent * 2.5
        self.viewer.cam.elevation = -40

    def camera2_setup(self):
        self.camera2.cam.trackbodyid = -1
        self.camera2.cam.distance = self.model.stat.extent * 2.5*1.4
        self.camera2.cam.elevation = -40

    def camera3_setup(self):
        self.camera3.cam.trackbodyid = -1
        self.camera3.cam.distance = self.model.stat.extent * 2.5*1.4
        self.camera3.cam.elevation = -40


class Box3dFixedReachEnvPixelGreyHarderTwoCam(mujoco_env.MujocoPixel2CamEnv, utils.EzPickle):
    def __init__(self, width=84, height=84):
        mujoco_env.MujocoPixel2CamEnv.__init__(self, 'arm_claw_new_one_box.xml', 4, width, height, "grey")
        utils.EzPickle.__init__(self)

        self.observation_space = Box(low=0, high=255, shape=(width, height, 2))


    def _step(self, a):
        self.do_simulation(a, self.frame_skip)
        
        obs = self._get_obs()
        done = False
        
        reach_reward = 0.0
        contact_reward = 0.0

        # Check for distance
        d = self.unwrapped.data
        distance = np.linalg.norm(d.site_xpos.flatten() - (list(d.qpos[-2:].flatten()) + [0.075]))
        # print d.qpos.flatten()
        # print distance
        if distance <= 0.1:
            reach_reward += 2.0 - distance*3
        reward = reach_reward

        return obs, reward, done, dict(reach_reward=reach_reward, contact_reward=contact_reward)


    def reset_model(self):
        c = 0.1
        qpos = self.init_qpos.copy()
        # qpos[:7] = [0.000, 3.133, 0.018, -1.500, -0.004, -0.000, 0.005]
        qpos[4:6] += self.np_random.uniform(low=-c, high=c, size=2)
        # qpos = self.init_qpos
        self.set_state(
            qpos,
            self.init_qvel,
        )
        return self._get_obs()
    
    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = self.model.stat.extent * 2.5
        self.viewer.cam.elevation = -40

    def camera2_setup(self):
        self.camera2.cam.trackbodyid = -1
        self.camera2.cam.distance = self.model.stat.extent * 2.5*1.4
        self.camera2.cam.elevation = -40

    def camera3_setup(self):
        self.camera3.cam.trackbodyid = -1
        self.camera3.cam.distance = self.model.stat.extent * 2.5*1.4
        self.camera3.cam.elevation = -40


class Box3dFixedReachEnvPixelGreyHarder(mujoco_env.MujocoPixelEnv, utils.EzPickle):
    def __init__(self, width=84, height=84):
        mujoco_env.MujocoPixelEnv.__init__(self, 'arm_claw_new.xml', 4, width, height, "grey")
        utils.EzPickle.__init__(self)

        self.observation_space = Box(low=0, high=255, shape=(width, height, 1))

    def _step(self, a):
        self.do_simulation(a, self.frame_skip)
        
        obs = self._get_obs()
        done = False
        
        reach_reward = 0.0
        contact_reward = 0.0

        # Check for distance
        d = self.unwrapped.data
        distance = np.linalg.norm(d.site_xpos.flatten() - (list(d.qpos[-2:].flatten()) + [0.075]))
        # print d.qpos.flatten()
        # print distance
        if distance <= 0.1:
            reach_reward += 2.0 - distance*3
        reward = reach_reward

        return obs, reward, done, dict(reach_reward=reach_reward, contact_reward=contact_reward)


    def reset_model(self):
        c = 0.1
        qpos = self.init_qpos.copy()
        qpos[:7] = [0.000, 3.133, 0.018, -1.500, -0.004, -0.000, 0.005]
        qpos[7:9] += self.np_random.uniform(low=-c, high=c, size=2)
        # qpos = self.init_qpos
        self.set_state(
            qpos,
            self.init_qvel,
        )
        return self._get_obs()
    
    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = self.model.stat.extent * 2.5
        self.viewer.cam.elevation = -40

    def camera2_setup(self):
        self.camera2.cam.trackbodyid = -1
        self.camera2.cam.distance = self.model.stat.extent * 2.5*1.4
        self.camera2.cam.elevation = -30
        self.camera2.cam.lookat[0] += 0.4
        self.camera2.cam.lookat[1] += 0.3
        self.camera2.cam.lookat[2] += 0

class Box3dFixedReachEnvPixelGreyHarderMulAct(mujoco_env.MujocoPixelEnv, utils.EzPickle):
    def __init__(self, width=84, height=84, action_mul=4):
        self.action_mul = action_mul
        mujoco_env.MujocoPixelEnv.__init__(self, 'arm_claw_new.xml', 4, width, height, "grey")
        utils.EzPickle.__init__(self)

        self.observation_space = Box(low=0, high=255, shape=(width, height, action_mul))

    def _step(self, a):
        obs_s = []
        for _ in range(self.action_mul):
            self.do_simulation(a, self.frame_skip)
            
            obs = self._get_obs()
            obs_s.append(obs)
        
        done = False
        
        reach_reward = 0.0
        contact_reward = 0.0

        # Check for distance
        d = self.unwrapped.data
        distance = np.linalg.norm(d.site_xpos.flatten() - (list(d.qpos[-2:].flatten()) + [0.075]))
        # print d.qpos.flatten()
        # print distance
        if distance <= 0.1:
            reach_reward += 2.0 - distance*3
        reward = reach_reward
        obs_return = np.concatenate(obs_s, axis=2)
        return obs_return, reward, done, dict(reach_reward=reach_reward, contact_reward=contact_reward)


    def reset_model(self):
        c = 0.1
        qpos = self.init_qpos.copy()
        qpos[:7] = [0.000, 3.133, 0.018, -1.500, -0.004, -0.000, 0.005]
        qpos[7:9] += self.np_random.uniform(low=-c, high=c, size=2)
        # qpos = self.init_qpos
        self.set_state(
            qpos,
            self.init_qvel,
        )
        obs = self._get_obs()
        return np.concatenate([obs, obs, obs, obs], axis=2)
    
    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = self.model.stat.extent * 2.5
        self.viewer.cam.elevation = -40

    def camera2_setup(self):
        self.camera2.cam.trackbodyid = -1
        self.camera2.cam.distance = self.model.stat.extent * 2.5*1.4
        self.camera2.cam.elevation = -30
        self.camera2.cam.lookat[0] += 0.4
        self.camera2.cam.lookat[1] += 0.3
        self.camera2.cam.lookat[2] += 0


class Box3dFixedReachEnvPixelGrey(mujoco_env.MujocoPixelEnv, utils.EzPickle):
    def __init__(self, width=84, height=84):
        mujoco_env.MujocoPixelEnv.__init__(self, 'arm_claw_new.xml', 4, width, height, "grey")
        utils.EzPickle.__init__(self)

        self.observation_space = Box(low=0, high=255, shape=(width, height, 1))

    def _step(self, a):
        self.do_simulation(a, self.frame_skip)
        
        obs = self._get_obs()
        done = False
        
        reach_reward = 0.0
        contact_reward = 0.0

        # Check for distance
        d = self.unwrapped.data
        distance = np.linalg.norm(d.site_xpos.flatten() - (list(d.qpos[-2:].flatten()) + [0.075]))
        # print d.qpos.flatten()
        # print distance
        if distance <= 0.4:
            reach_reward += 2.0 - distance*3
        reward = reach_reward

        return obs, reward, done, dict(reach_reward=reach_reward, contact_reward=contact_reward)


    def reset_model(self):
        c = 0.1
        qpos = self.init_qpos.copy()
        qpos[:7] = [0.000, 3.133, 0.018, -1.500, -0.004, -0.000, 0.005]
        qpos[7:9] += self.np_random.uniform(low=-c, high=c, size=2)
        # qpos = self.init_qpos
        self.set_state(
            qpos,
            self.init_qvel,
        )
        return self._get_obs()
    
    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = self.model.stat.extent * 2.5
        self.viewer.cam.elevation = -40

    def camera2_setup(self):
        self.camera2.cam.trackbodyid = -1
        self.camera2.cam.distance = self.model.stat.extent * 2.5*1.4
        self.camera2.cam.elevation = -30
        self.camera2.cam.lookat[0] += 0.4
        self.camera2.cam.lookat[1] += 0.3
        self.camera2.cam.lookat[2] += 0
        
        
class Box3dMulMulObjConAvoidPixelGreyEnv(mujoco_env.MujocoPixelEnv, utils.EzPickle):
    def __init__(self, width=84, height=84):
        mujoco_env.MujocoPixelEnv.__init__(self, 'arm_reach_mul_mul.xml', 4, width, height, "grey")
        utils.EzPickle.__init__(self)

    def check_contact(self):
        d = self.unwrapped.data
        for coni in range(d.ncon):
            con = d.obj.contact[coni]
            if  (con.geom1 != 0 and \
                 con.geom1 != 4 and \
                 con.geom1 != 5 and \
                 con.geom1 != 6 and \
                 con.geom1 != 7 and \
                 con.geom1 != 8 and \
                 con.geom1 != 9) \
            and (con.geom2 == 4 or \
                 con.geom2 == 5 or \
                 con.geom2 == 6 or \
                 con.geom2 == 7 or \
                 con.geom2 == 8 or \
                 con.geom2 == 9):
                # Small box is touched but not by table
                return True
        return False

    def check_obj_contact(self):
        d = self.unwrapped.data
        for coni in range(d.ncon):
            con = d.obj.contact[coni]
            if  (con.geom1 == 4 or \
                 con.geom1 == 5 or \
                 con.geom1 == 6 or \
                 con.geom1 == 7 or \
                 con.geom1 == 8 or \
                 con.geom1 == 9) \
            and (con.geom2 == 4 or \
                 con.geom2 == 5 or \
                 con.geom2 == 6 or \
                 con.geom2 == 7 or \
                 con.geom2 == 8 or \
                 con.geom2 == 9):
                # Small box is touched but not by table
                return True
        return False

    def _step(self, a):
        self.do_simulation(a, self.frame_skip)
        
        obs = self._get_obs()
        done = False
        
        contact_reward = 0.0

        # Check for distance
        if self.check_contact():
            contact_reward = 1.0

        reward = 0.0

        return obs, reward, done, dict(contact_reward=contact_reward)

    def reset_model(self):
        c = 0.2
        # qpos = [0.000, 3.133, 0.018, -1.500, -0.004, -0.000, 0.005, -0.001, 0.007]
        while True:
            qpos = self.init_qpos.copy()
            qpos[:7] = [0.000, 3.133, 0.018, -1.500, -0.004, -0.000, 0.005] + self.np_random.uniform(low=-0.1, high=0.1, size=7)
            qpos[7:9] += self.np_random.uniform(low=-c, high=c, size=2)
            qpos[14:16] += self.np_random.uniform(low=-c, high=c, size=2)
            qpos[21:23] += self.np_random.uniform(low=-c, high=c, size=2)
            qpos[28:30] += self.np_random.uniform(low=-c, high=c, size=2)
            qpos[35:37] += self.np_random.uniform(low=-c, high=c, size=2)
            qpos[42:44] += self.np_random.uniform(low=-c, high=c, size=2)

            self.set_state(
                np.array(qpos),
                self.init_qvel,
            )

            if not self.check_obj_contact():
                break
        return self._get_obs()


    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = self.model.stat.extent * 2.5
        self.viewer.cam.elevation = -40

    def camera2_setup(self):
        self.camera2.cam.trackbodyid = -1
        self.camera2.cam.distance = self.model.stat.extent * 2.5*1.4
        self.camera2.cam.elevation = -30
        self.camera2.cam.lookat[0] += 0.4
        self.camera2.cam.lookat[1] += 0.3
        self.camera2.cam.lookat[2] += 0



class Box3dFixedReachEnvPixelRGB(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, width=42, height=42):
        self.width = width
        self.height = height

        mujoco_env.MujocoEnv.__init__(self, 'arm_claw_new.xml', 4)
        utils.EzPickle.__init__(self)

        self.observation_space = Box(low=0, high=255, shape=(3, width, height))

    def _step(self, a):
        self.do_simulation(a, self.frame_skip)
        
        obs = self._get_obs()
        done = False
        
        reach_reward = 0.0
        contact_reward = 0.0

        # Check for distance
        d = self.unwrapped.data
        distance = np.linalg.norm(d.site_xpos.flatten() - (list(d.qpos[-2:].flatten()) + [0.075]))
        # print distance
        if distance <= 0.4:
            reach_reward += 2.0 - distance*3
        reward = reach_reward

        return obs, reward, done, dict(reach_reward=reach_reward, contact_reward=contact_reward)


    def _get_obs(self):
        # camera2_output = None
        # self.camera2.render()
        # data, width, height = self.camera2.get_image()
        # camera2_output =  np.fromstring(data, dtype='uint8').reshape(height, width, 3)[::-1, :, :]
        # return np.concatenate((self._render(mode="rgb_array"), camera2_output), axis = 2)
        obs = self._render(mode="rgb_array")
        return np.transpose(obs, axes=(2,1,0))

    def reset_model(self):
        c = 0.01
        qpos = [0.000, 3.133, 0.018, -1.500, -0.004, -0.000, 0.005, -0.001, 0.007]
        qpos[:7] += self.np_random.uniform(low=-c, high=c, size=7)
        self.set_state(
            qpos,
            self.init_qvel,
        )
        return self._get_obs()
    
    def _get_viewer(self):
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer(init_width=self.width, init_height=self.height)
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





class Box3dFixedReachMulObjEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'arm_reach_mul.xml', 4)
        utils.EzPickle.__init__(self)
        

    def _step(self, a):
        self.do_simulation(a, self.frame_skip)
        
        obs = self._get_obs()
        done = False
        
        contact_reward = 0.0

        # Check for distance
        d = self.unwrapped.data
        for coni in range(d.ncon):
            con = d.obj.contact[coni]
            if con.geom1 != 0 and (con.geom2 == 4 or \
                                   con.geom2 == 5 or \
                                   con.geom2 == 6):
                # Small box is touched but not by table
                contact_reward = 1.0
        # print (d.site_xpos.flatten())
        # distance = np.linalg.norm(d.site_xpos.flatten() - (list(d.qpos[-2:].flatten()) + [0.025]))
        # if distance <= 0.05:
        #     reach_reward += 1.0 - distance*3

        reward = 0.0

        return obs, reward, done, dict(contact_reward=contact_reward)


    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat,
            self.model.data.qvel.flat
        ])

    def reset_model(self):
        c = 0.1
        # qpos = [0.000, 3.133, 0.018, -1.500, -0.004, -0.000, 0.005, -0.001, 0.007]
        qpos = self.init_qpos.copy()

        qpos[1] += 1.0
        qpos[:7] += self.np_random.uniform(low=-0.1, high=0.1, size=7)
        qpos[7:9] += self.np_random.uniform(low=-c, high=c, size=2)
        qpos[14:16] += self.np_random.uniform(low=-c, high=c, size=2)
        qpos[21:23] += self.np_random.uniform(low=-c, high=c, size=2)
        
        self.set_state(
            np.array(qpos),
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




class Box3dFixedReachMulObjConAvoidEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'arm_reach_mul.xml', 4)
        utils.EzPickle.__init__(self)
        

    def check_contact(self):
        d = self.unwrapped.data
        for coni in range(d.ncon):
            con = d.obj.contact[coni]
            if con.geom1 != 0 \
            and con.geom1 != 4 \
            and con.geom1 != 5 \
            and con.geom1 != 6 \
            and (con.geom2 == 4 or \
                 con.geom2 == 5 or \
                 con.geom2 == 6):
                # Small box is touched but not by table
                return True
        return False

    def check_obj_contact(self):
        d = self.unwrapped.data
        for coni in range(d.ncon):
            con = d.obj.contact[coni]
            if (con.geom1 == 4 or \
                 con.geom1 == 5 or \
                 con.geom1 == 6) \
            and (con.geom2 == 4 or \
                 con.geom2 == 5 or \
                 con.geom2 == 6):
                # Small box is touched but not by table
                return True
        return False

    def _step(self, a):
        self.do_simulation(a, self.frame_skip)
        
        obs = self._get_obs()
        done = False
        
        contact_reward = 0.0

        # Check for distance
        if self.check_contact():
            contact_reward = 1.0

        reward = 0.0

        return obs, reward, done, dict(contact_reward=contact_reward)

    def reset_model(self):
        c = 0.1
        # qpos = [0.000, 3.133, 0.018, -1.500, -0.004, -0.000, 0.005, -0.001, 0.007]
        while True:
            qpos = self.init_qpos.copy()

            qpos[1] += 1.0
            qpos[:7] += self.np_random.uniform(low=-0.1, high=0.1, size=7)
            qpos[7:9] += self.np_random.uniform(low=-c, high=c, size=2)
            qpos[14:16] += self.np_random.uniform(low=-c, high=c, size=2)
            qpos[21:23] += self.np_random.uniform(low=-c, high=c, size=2)
            
            self.set_state(
                np.array(qpos),
                self.init_qvel,
            )

            if not self.check_obj_contact():
                break
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




class Box3dFixedReachMulMulObjConAvoidEnvLimit(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'arm_reach_mul_mul_limit.xml', 4)
        utils.EzPickle.__init__(self)
        
    def check_contact(self):
        d = self.unwrapped.data
        for coni in range(d.ncon):
            con = d.obj.contact[coni]
            if  (con.geom1 != 0 and \
                 con.geom1 != 4 and \
                 con.geom1 != 5 and \
                 con.geom1 != 6 and \
                 con.geom1 != 7 and \
                 con.geom1 != 8 and \
                 con.geom1 != 9) \
            and (con.geom2 == 4 or \
                 con.geom2 == 5 or \
                 con.geom2 == 6 or \
                 con.geom2 == 7 or \
                 con.geom2 == 8 or \
                 con.geom2 == 9):
                # Small box is touched but not by table
                return True
        return False

    def check_obj_contact(self):
        d = self.unwrapped.data
        for coni in range(d.ncon):
            con = d.obj.contact[coni]
            if  (con.geom1 == 4 or \
                 con.geom1 == 5 or \
                 con.geom1 == 6 or \
                 con.geom1 == 7 or \
                 con.geom1 == 8 or \
                 con.geom1 == 9) \
            and (con.geom2 == 4 or \
                 con.geom2 == 5 or \
                 con.geom2 == 6 or \
                 con.geom2 == 7 or \
                 con.geom2 == 8 or \
                 con.geom2 == 9):
                # Small box is touched but not by table
                return True
        return False

    def _step(self, a):
        self.do_simulation(a, self.frame_skip)
        
        obs = self._get_obs()
        done = False
        
        contact_reward = 0.0

        # Check for distance
        if self.check_contact():
            contact_reward = 1.0

        reward = 0.0

        return obs, reward, done, dict(contact_reward=contact_reward)

    def reset_model(self):
        c = 0.2
        # qpos = [0.000, 3.133, 0.018, -1.500, -0.004, -0.000, 0.005, -0.001, 0.007]
        while True:
            qpos = self.init_qpos.copy()

            qpos[1] += 1.0
            qpos[:7] += self.np_random.uniform(low=-0.1, high=0.1, size=7)
            qpos[7:9] += self.np_random.uniform(low=-c, high=c, size=2)
            qpos[14:16] += self.np_random.uniform(low=-c, high=c, size=2)
            qpos[21:23] += self.np_random.uniform(low=-c, high=c, size=2)
            qpos[28:30] += self.np_random.uniform(low=-c, high=c, size=2)
            qpos[35:37] += self.np_random.uniform(low=-c, high=c, size=2)
            qpos[42:44] += self.np_random.uniform(low=-c, high=c, size=2)

            self.set_state(
                np.array(qpos),
                self.init_qvel,
            )

            if not self.check_obj_contact():
                break
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



class Box3dFixedReachMulMulObjConAvoidEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'arm_reach_mul_mul.xml', 4)
        utils.EzPickle.__init__(self)
        
    def check_contact(self):
        d = self.unwrapped.data
        for coni in range(d.ncon):
            con = d.obj.contact[coni]
            if  (con.geom1 != 0 and \
                 con.geom1 != 3 and \
                 con.geom1 != 4 and \
                 con.geom1 != 5 and \
                 con.geom1 != 6 and \
                 con.geom1 != 7 and \
                 con.geom1 != 8) \
            and (con.geom2 == 3 or \
                 con.geom2 == 4 or \
                 con.geom2 == 5 or \
                 con.geom2 == 6 or \
                 con.geom2 == 7 or \
                 con.geom2 == 8):
                # Small box is touched but not by table
                return True
        return False

    def check_obj_contact(self):
        d = self.unwrapped.data
        for coni in range(d.ncon):
            con = d.obj.contact[coni]
            if  (con.geom1 == 3 or \
                 con.geom1 == 4 or \
                 con.geom1 == 5 or \
                 con.geom1 == 6 or \
                 con.geom1 == 7 or \
                 con.geom1 == 8) \
            and (con.geom2 == 3 or \
                 con.geom2 == 4 or \
                 con.geom2 == 5 or \
                 con.geom2 == 6 or \
                 con.geom2 == 7 or \
                 con.geom2 == 8):
                # Small box is touched but not by table
                return True
        return False

    def check_table_contact(self):
        d = self.unwrapped.data
        for coni in range(d.ncon):
            con = d.obj.contact[coni]
            if  (con.geom1 == 0) \
            and (con.geom2 == 1 or con.geom2 == 2):
                # Small box is touched but not by table
                return True
        return False

    def _step(self, a):
        self.do_simulation(a, self.frame_skip)
        
        obs = self._get_obs()
        done = False
        
        contact_reward = 0.0
        table_contact_reward = 0.0

        # Check for distBox3dFixedReachMulMulObjConAvoidEnvance
        if self.check_contact():
            contact_reward = 1.0
        if self.check_table_contact():
            table_contact_reward = 1.0


        reward = contact_reward

        return obs, reward, done, dict(contact_reward=contact_reward, table_contact_reward=table_contact_reward)

    def reset_model(self):
        c = 0.4
        # qpos = [0.000, 3.133, 0.018, -1.500, -0.004, -0.000, 0.005, -0.001, 0.007]
        while True:
            qpos = self.init_qpos.copy()

            qpos[1] += 1.0
            qpos[:5] += self.np_random.uniform(low=-0.1, high=0.1, size=5)
            qpos[5:7] += self.np_random.uniform(low=-c, high=c, size=2)
            qpos[12:14] += self.np_random.uniform(low=-c, high=c, size=2)
            qpos[19:21] += self.np_random.uniform(low=-c, high=c, size=2)
            qpos[26:28] += self.np_random.uniform(low=-c, high=c, size=2)
            qpos[33:35] += self.np_random.uniform(low=-c, high=c, size=2)
            qpos[40:42] += self.np_random.uniform(low=-c, high=c, size=2)

            self.set_state(
                np.array(qpos),
                self.init_qvel,
            )

            if not self.check_obj_contact():
                break
        return self._get_obs()

    def _get_obs(self):
        # return self.model.data.qpos.flat[:9]
        return np.concatenate([
           self.model.data.qpos.flat,
           self.model.data.qvel.flat
        ])

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = self.model.stat.extent * 2.2
        self.viewer.cam.elevation = -40
        self.viewer.cam.lookat[0] += 0.2
        self.viewer.cam.lookat[1] -= 0.1



class Box3dFixedReachMulObjConAvoidMoreEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'arm_reach_mul.xml', 4)
        utils.EzPickle.__init__(self)
        
    def check_reachable(self):
        o = [0.15, 0]
        qpos = self.init_qpos.copy()
        dis1 = np.linalg.norm(qpos[7:9]-o)
        dis2 = np.linalg.norm(qpos[14:16]-o)
        dis3 = np.linalg.norm(qpos[21:23]-o)
        dis = np.array([dis1, dis2, dis3])
        return all(dis >= 0.55)

    def check_contact(self):
        d = self.unwrapped.data
        for coni in range(d.ncon):
            con = d.obj.contact[coni]
            if con.geom1 != 0 \
            and con.geom1 != 4 \
            and con.geom1 != 5 \
            and con.geom1 != 6 \
            and (con.geom2 == 4 or \
                 con.geom2 == 5 or \
                 con.geom2 == 6):
                # Small box is touched but not by table
                return True
        return False

    def check_obj_contact(self):
        d = self.unwrapped.data
        for coni in range(d.ncon):
            con = d.obj.contact[coni]
            if (con.geom1 == 4 or \
                 con.geom1 == 5 or \
                 con.geom1 == 6) \
            and (con.geom2 == 4 or \
                 con.geom2 == 5 or \
                 con.geom2 == 6):
                # Small box is touched but not by table
                return True
        return False

    def _step(self, a):
        self.do_simulation(a, self.frame_skip)
        
        obs = self._get_obs()
        done = False
        
        contact_reward = 0.0

        # Check for distance
        if self.check_contact():
            contact_reward = 1.0

        reward = 0.0

        return obs, reward, done, dict(contact_reward=contact_reward)

    def reset_model(self):
        c = 0.2
        # qpos = [0.000, 3.133, 0.018, -1.500, -0.004, -0.000, 0.005, -0.001, 0.007]
        while True:
            qpos = self.init_qpos.copy()

            qpos[1] += 1.0
            qpos[:7] += self.np_random.uniform(low=-0.1, high=0.1, size=7)
            qpos[7:9] += self.np_random.uniform(low=-c, high=c, size=2)
            qpos[14:16] += self.np_random.uniform(low=-c, high=c, size=2)
            qpos[21:23] += self.np_random.uniform(low=-c, high=c, size=2)
            
            self.set_state(
                np.array(qpos),
                self.init_qvel,
            )

            if not self.check_obj_contact():
            # if True:
                break
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

class Box3dFixedReachMulObjPrevVelEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'arm_reach_mul.xml', 4)
        utils.EzPickle.__init__(self)
        self.prev_vel = self.init_qvel.copy()


    def _step(self, a):
        self.prev_vel = self.model.data.qvel.copy()
        self.do_simulation(a, self.frame_skip)
        
        obs = self._get_obs()
        done = False
        
        contact_reward = 0.0

        # Check for distance
        d = self.unwrapped.data
        for coni in range(d.ncon):
            con = d.obj.contact[coni]
            if con.geom1 != 0 and (con.geom2 == 4 or \
                                   con.geom2 == 5 or \
                                   con.geom2 == 6):
                # Small box is touched but not by table
                contact_reward = 1.0
        # print (d.site_xpos.flatten())
        # distance = np.linalg.norm(d.site_xpos.flatten() - (list(d.qpos[-2:].flatten()) + [0.025]))
        # if distance <= 0.05:
        #     reach_reward += 1.0 - distance*3

        reward = 0.0

        return obs, reward, done, dict(contact_reward=contact_reward)

    def reset_model(self):
        c = 0.1
        # qpos = [0.000, 3.133, 0.018, -1.500, -0.004, -0.000, 0.005, -0.001, 0.007]
        qpos = self.init_qpos.copy()

        qpos[1] += 1.0
        qpos[:7] += self.np_random.uniform(low=-0.1, high=0.1, size=7)
        qpos[7:9] += self.np_random.uniform(low=-c, high=c, size=2)
        qpos[14:16] += self.np_random.uniform(low=-c, high=c, size=2)
        qpos[21:23] += self.np_random.uniform(low=-c, high=c, size=2)
        
        self.set_state(
            np.array(qpos),
            self.init_qvel,
        )
        self.prev_vel = self.init_qvel.copy()
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
           self.model.data.qpos.flat,
           self.model.data.qvel.flat,
           self.prev_vel.flat
        ])

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = self.model.stat.extent * 2.5
        self.viewer.cam.elevation = -40

class Box3dFixedReachHardestEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'arm_claw_v5.xml', 4)
        utils.EzPickle.__init__(self)
        

    def _step(self, a):
        self.do_simulation(a, self.frame_skip)
        
        obs = self._get_obs()
        done = False
        
        reach_reward = 0.0
        contact_reward = 0.0

        # Check for distance
        d = self.unwrapped.data
        
        distance = np.linalg.norm(d.site_xpos.flatten() - (list(d.qpos[-2:].flatten()) + [0.075]))
        if distance <= 0.05:
            reach_reward += 1.0 - distance*3

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
        qpos[:7] += self.np_random.uniform(low=-c, high=c, size=7)
        qpos[-2:] += self.np_random.uniform(low=-0.1, high=0.1, size=2)
        self.set_state(
            np.array(qpos),
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
