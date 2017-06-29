from gym.envs.mujoco.mujoco_env import MujocoEnv
# ^^^^^ so that user gets the correct error
# message if mujoco is not installed correctly
from gym.envs.mujoco.ant import AntEnv
from gym.envs.mujoco.half_cheetah import HalfCheetahEnv
from gym.envs.mujoco.hopper import HopperEnv
from gym.envs.mujoco.walker2d import Walker2dEnv
from gym.envs.mujoco.humanoid import HumanoidEnv
from gym.envs.mujoco.inverted_pendulum import InvertedPendulumEnv
from gym.envs.mujoco.inverted_double_pendulum import InvertedDoublePendulumEnv
from gym.envs.mujoco.reacher import ReacherEnv
from gym.envs.mujoco.swimmer import SwimmerEnv
from gym.envs.mujoco.humanoidstandup import HumanoidStandupEnv

# Custom environment
from gym.envs.mujoco.box3d import Box3dFixedReachEnvPixelGrey, \
								  Box3dFixedReachEnvPixelGreyHarder, \
								  Box3dFixedReachEnvPixelGreyHarderTwoCam, \
								  Box3dFixedReachEnvPixelGreyHarderTwoCamMulAct, \
								  Box3dFixedReachEnvPixelGreyMulMulContactTwoCamMulAct, \
								  Box3dFixedReachEnvPixelGreyMulMulContactTwoCamMulActNoMas, \
								  Box3dFixedReachEnvPixelGreyMulMulContactTwoCamMulActLess, \
								  Box3dFixedReachEnvPixelGreyMulMulContactTwoCamMulActFusion, \
								  Box3dFixedReachEnvPixelGreyHarderTwoCamMulActLess, \
								  Box3dFixedReachEnvPixelGreyHarderTwoCamMulActLessRepeatTwo, \
								  Box3dFixedReachEnvPixelGreyMulMulTwoCamMulAct, \
								  Box3dFixedReachEnvPixelGreyMulMulTwoCamMulActLess, \
								  Box3dFixedReachEnvPixelGreyHarderMulAct, \
								  Box3dFixedReachEnvPixelRGB, \
								  Box3dReachEnv, Box3dGraspEnv, \
								  Box3dReachPosEnv, \
								  Box3dFixedReachEnv, \
								  Box3dNoRewardEnv, \
								  Box3dContactReachEnv, \
								  Box3dFixedReachHarderEnv, \
								  Box3dFixedReachHardestEnv, \
								  Box3dFixedReachMulObjEnv, \
								  Box3dFixedReachMulObjConAvoidEnv, \
								  Box3dFixedReachMulObjPrevVelEnv, \
								  Box3dFixedReachMulObjConAvoidMoreEnv, \
								  Box3dFixedReachMulMulObjConAvoidEnv, \
								  Box3dFixedReachMulMulObjConAvoidEnvOne, \
								  Box3dMulMulObjConAvoidPixelGreyEnv
