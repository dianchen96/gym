import os

from gym import error, spaces
from gym.utils import seeding
import numpy as np
from os import path
import gym
import six

try:
    import mujoco_py
    from mujoco_py.mjlib import mjlib
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))

class MujocoEnv(gym.Env):
    """Superclass for all MuJoCo environments.
    """

    def __init__(self, model_path, frame_skip):
        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
        if not path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)
        self.frame_skip = frame_skip
        self.model = mujoco_py.MjModel(fullpath)
        self.data = self.model.data
        self.viewer = None
        
        # self.camera2 = None
        # #import pdb; pdb.set_trace()
        # self.camera2 = mujoco_py.MjViewer(init_width=500, init_height=500)
        # self.camera2.start()
        # self.camera2.set_model(self.model)
        # self.camera2_setup()
        
        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.init_qpos = self.model.data.qpos.ravel().copy()
        self.init_qvel = self.model.data.qvel.ravel().copy()
        observation, _reward, done, _info = self._step(np.zeros(self.model.nu))
        assert not done
        self.obs_dim = observation.size

        bounds = self.model.actuator_ctrlrange.copy()
        low = bounds[:, 0]
        high = bounds[:, 1]
        self.action_space = spaces.Box(low, high)

        high = np.inf*np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high)

        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # methods to override:
    # ----------------------------

    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        raise NotImplementedError

    def viewer_setup(self):
        """
        This method is called when the viewer is initialized and after every reset
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        pass

    # -----------------------------

    def _reset(self):
        mjlib.mj_resetData(self.model.ptr, self.data.ptr)
        ob = self.reset_model()
        if self.viewer is not None:
            self.viewer.autoscale()
            self.viewer_setup()
        return ob

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        self.model.data.qpos = qpos
        self.model.data.qvel = qvel
        self.model._compute_subtree()  # pylint: disable=W0212
        self.model.forward()

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames):
        self.model.data.ctrl = ctrl
        for _ in range(n_frames):
            self.model.step()

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self._get_viewer().finish()
                self.viewer = None
            return
        if mode == 'rgb_array':
            self._get_viewer().render()
            data, width, height = self._get_viewer().get_image()
            return np.fromstring(data, dtype='uint8').reshape(height, width, 3)[::-1, :, :]
        elif mode == 'human':
            self._get_viewer().loop_once()

    def _get_viewer(self):
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer()
            self.viewer.start()
            self.viewer.set_model(self.model)
            self.viewer_setup()
        return self.viewer

    def get_body_com(self, body_name):
        idx = self.model.body_names.index(six.b(body_name))
        return self.model.data.com_subtree[idx]

    def get_body_comvel(self, body_name):
        idx = self.model.body_names.index(six.b(body_name))
        return self.model.body_comvels[idx]

    def get_body_xmat(self, body_name):
        idx = self.model.body_names.index(six.b(body_name))
        return self.model.data.xmat[idx].reshape((3, 3))

    def state_vector(self):
        return np.concatenate([
            self.model.data.qpos.flat,
            self.model.data.qvel.flat
        ])
 

class MujocoPixelEnv(MujocoEnv):
    def __init__(
        self, 
        model_path, 
        frame_skip, 
        width=42,
        height=42,
        mode="rgb"
    ):
        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
        if not path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)
        self.frame_skip = frame_skip
        self.model = mujoco_py.MjModel(fullpath)
        self.data = self.model.data
        self.width = width
        self.height = height
        self.mode = mode

        self.viewer = None
        
        self.camera2 = None
        self.camera2 = mujoco_py.MjViewer(init_width=self.width, init_height=self.height)
        self.camera2.start()
        self.camera2.set_model(self.model)
        self.camera2_setup()
        
        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.init_qpos = self.model.data.qpos.ravel().copy()
        self.init_qvel = self.model.data.qvel.ravel().copy()
        observation, _reward, done, _info = self._step(np.zeros(self.model.nu))
        assert not done
        self.obs_dim = observation.size

        bounds = self.model.actuator_ctrlrange.copy()
        low = bounds[:, 0]
        high = bounds[:, 1]
        self.action_space = spaces.Box(low, high)

        high = np.inf*np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high)

        self._seed()


    def camera2_setup(self):
        raise NotImplementedError

    def _get_obs(self):
        camera2_output = None
        self.camera2.render()
        data, width, height = self.camera2.get_image()
        camera2_output = np.fromstring(data, dtype='uint8').reshape(height, width, 3)[::-1, :, :]
        if self.mode == "grey":
            camera2_output = np.mean(camera2_output, axis=2)[:, :, np.newaxis]
        return camera2_output


class MujocoPixel2CamEnv(MujocoEnv):
    def __init__(
        self, 
        model_path, 
        frame_skip, 
        width=42,
        height=42,
        mode="rgb"
    ):
        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
        if not path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)
        self.frame_skip = frame_skip
        self.model = mujoco_py.MjModel(fullpath)
        self.data = self.model.data
        self.width = width
        self.height = height
        self.mode = mode

        self.viewer = None
        
        self.camera2 = None
        self.camera2 = mujoco_py.MjViewer(init_width=self.width, init_height=self.height)
        self.camera2.start()
        self.camera2.set_model(self.model)
        self.camera2_setup()

        self.camera3 = None
        self.camera3 = mujoco_py.MjViewer(init_width=self.width, init_height=self.height)
        self.camera3.start()
        self.camera3.set_model(self.model)
        self.camera3_setup()

        azimuth = self.camera2.cam.azimuth
        self.camera3.cam.azimuth = azimuth + 180
        
        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.init_qpos = self.model.data.qpos.ravel().copy()
        self.init_qvel = self.model.data.qvel.ravel().copy()
        observation, _reward, done, _info = self._step(np.zeros(self.model.nu))
        assert not done
        self.obs_dim = observation.size

        bounds = self.model.actuator_ctrlrange.copy()
        low = bounds[:, 0]
        high = bounds[:, 1]
        self.action_space = spaces.Box(low, high)

        high = np.inf*np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high)

        self._seed()


    def camera2_setup(self):
        raise NotImplementedError

    def camera3_setup(self):
        raise NotImplementedError

    def _get_obs(self):
        camera2_output = None
        self.camera2.render()
        data, width, height = self.camera2.get_image()
        camera2_output = np.fromstring(data, dtype='uint8').reshape(height, width, 3)[::-1, :, :]
        if self.mode == "grey":
            camera2_output = np.mean(camera2_output, axis=2)[:, :, np.newaxis]

        camera3_output = None
        self.camera3.render()
        data, width, height = self.camera3.get_image()
        camera3_output = np.fromstring(data, dtype='uint8').reshape(height, width, 3)[::-1, :, :]
        if self.mode == "grey":
            camera3_output = np.mean(camera3_output, axis=2)[:, :, np.newaxis]
        
        return np.concatenate([camera2_output, camera3_output], axis=2)
