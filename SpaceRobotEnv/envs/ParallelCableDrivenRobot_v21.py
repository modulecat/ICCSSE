import os
import gym
import mujoco_py
import numpy as np

from gym import spaces
from gym.utils import seeding

from gym.envs.mujoco.mujoco_env import convert_observation_to_space
from gym.envs.robotics import rotations

MODEL_XML_PATH = os.path.join(fr"D:\Research\ICCSSE\ParalllelCableDrivenRobot\SpaceRobotEnv\assets", "ParallelCableDrivenRobot_v2.xml")


class ParallelCableDrivenRobot_v21(gym.Env):
    def __init__(self, frame_skip=4):
        self.frame_skip = frame_skip
        self.model = mujoco_py.load_model_from_path(MODEL_XML_PATH)
        self.sim = mujoco_py.MjSim(self.model)
        self.data = self.sim.data
        self.viewer = None
        self._viewers = {}
        self.stop_distance = 0.05
        self.seed()

        self.metadata = {
            "render.modes": ["human"],
            "video.frames_per_second": int(np.round(1.0 / self.dt)),
        }

        self._set_action_space()
        action = self.action_space.sample()
        observation, _reward, done, _info = self.step(action)
        self._set_observation_space(observation)
        self.init_qpos = self.data.qpos
        self.init_qvel = self.data.qvel
        
        
    def _set_action_space(self):
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    def _set_observation_space(self, observation):
        self.observation_space = convert_observation_to_space(observation)
        return self.observation_space

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):

        self.do_simulation(action, self.frame_skip)
        current_info = self._get_current_info()
        obs = self._get_obs(current_info)
        reward = self._get_reward(current_info, action)
        done = self._get_terminal(current_info)

        return obs, reward, done, current_info
    def get_observation(self):
        current_info = self._get_current_info()
        obs = self._get_obs(current_info)
        return obs
        
    def reset(self):
        self.sim.reset()
        ob = self.reset_model()
        return ob

    def close(self):
        if self.viewer is not None:
            # self.viewer.finish()
            self.viewer = None
            self._viewers = {}

    def render(self, mode="human"):
        self._get_viewer(mode).render()

    def reset_model(self):
        
        # flag = (-1)**np.random.randint(2)
        # self.sim.model.body_pos[self.sim.model.body_name2id('object_point')] = \
        #     np.array([2.5, 1.0 * flag, 0]) + np.array([
        #         0.2 * np.random.random() - 0.1, (0.2 * np.random.random() - 0.1)*flag,
        #         0.2 * np.random.random() - 0.1
        #     ])
        self.sim.model.body_pos[self.sim.model.body_name2id('object_point')] = np.array([1.78, 0, 0.1])
        self.sim.forward()
        return self._get_obs(self._get_current_info())
    
    # 用于为各关节赋予初始位置与初始速度
    def set_state(self,qpos,qvel):
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(
            old_state.time, qpos, qvel, old_state.act, old_state.udd_state
        )
        self.sim.set_state(new_state)
        self.sim.forward()
        
    # agent.select_action()输出的action位于(-1,1)中, 本函数用于将action等比例恢复为环境的action_space对应区间
    def action_scaling(self,action):
        act_k=(self.action_space.high-self.action_space.low)/2.0
        act_b=(self.action_space.high+self.action_space.low)/2.0
        action=action*act_k+act_b
        # print(f'act_k={act_k}\nact_b={act_b}\naction={action}')
        return action
    
    def do_simulation(self, ctrl, n_frames):
        self.sim.data.ctrl[:len(ctrl)] = ctrl
        for _ in range(n_frames):
            self.sim.step()

    def _detecte_collision(self):
        self.collision = self.sim.data.ncon
        return self.collision


    def _get_current_info(self):
        # base position and velocity
        base_pos = np.concatenate((self.sim.data.get_body_xpos('base_body'), # position 3
                                   self.sim.data.get_body_xquat('base_body'))) # attitude 4
        base_vel = np.concatenate((self.sim.data.get_body_xvelp('base_body'), # pos velocity 3
                                   self.sim.data.get_body_xvelr('base_body'))) # rotate velocity 3

        # end-effector position and velocity
        end_pos = self.sim.data.get_site_xpos('point_end') # position 3
        end_vel = self.sim.data.get_site_xvelp('point_end') # pos velocity 3

        # target position and velocity
        target_pos = self.sim.data.get_site_xpos('object_point') # position 3
        # target_vel = self.sim.data.get_site_xvelp('object_point') # pos velocity 3


        # distance between end effector and target
        distance = np.linalg.norm(end_pos - target_pos)

        return {
            "end_effector_position": end_pos, "end_effector_velocity": end_vel,
            "target_position": target_pos,
            "distance": np.array([distance]),
            "base_pose": base_pos, "base_velocity": base_vel
        }

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == "human":
                self.viewer = mujoco_py.MjViewer(self.sim)
            elif mode == "rgb_array" or mode == "depth_array":
                assert "mode is not supported!"

            # self.viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer

    def _get_obs(self, current_info):
        return np.concatenate([
            current_info["end_effector_position"], current_info["end_effector_velocity"],
            current_info["target_position"], current_info["distance"],
            current_info["base_pose"], current_info["base_velocity"]
        ])

    def _get_reward(self, current_info, action):
        end_effector_position=current_info["end_effector_position"]
        end_effector_velocity = current_info["end_effector_velocity"]
        target_position=current_info["target_position"]
        distance = current_info["distance"][0]
        base_velocity = current_info["base_velocity"]
        base_position = current_info["base_pose"]

        # 显示当前状态,动作信息
        # print('nPe={},nVe={},nPb={},nVb={},nPt={},Dis={},nA={}'.format(np.linalg.norm(end_effector_position),\
        #                                                         np.linalg.norm(end_effector_velocity),\
        #                                                         np.linalg.norm(base_position),\
        #                                                         np.linalg.norm(base_velocity),\
        #                                                         np.linalg.norm(target_position),\
        #                                                         distance,\
        #                                                         np.linalg.norm(action)))


        # # success 1
        # reward = - distance - np.log(distance + 1e-6)
        # success 2
        # reward = - distance - np.log(distance + 1e-6) \
        #          - 0.01 * (np.linalg.norm(end_effector_velocity)+ np.linalg.norm(base_velocity)) \
        #          - 0.01 * np.linalg.norm(action)

        # #
        # reward = - 2.5 * distance - 0.5 * np.log(distance + 1e-6)\
        #          - 0.01 * (np.linalg.norm(end_effector_velocity)+ np.linalg.norm(base_velocity)) # 2

        # reward = - 3 * distance - 0.8 * np.log(distance + 1e-6)\
        #          - 0.01 * (np.linalg.norm(end_effector_velocity)+ np.linalg.norm(base_velocity))

        # if distance <= 0.06 and bool(self._detecte_collision()):
        #     reward -= 20
        # reward of 0225
        # reward = - 2*distance - np.log(distance + 1e-6) \
        #          - 0.02 * np.linalg.norm(action)
                 
        # reward of 0225_1
        # reward = - 5*distance - np.log(distance + 1e-6) \
        #          - 0.02 * np.linalg.norm(action)
        # reward of 0225_2
        # reward = - 5*distance - np.log(distance + 1e-6) \
        #          - 0.002 * np.linalg.norm(action)
        # reward of 0225_3
        # reward of 0517_0
        # reward = - 2*distance - np.log(distance + 1e-6)
        # reward of 0517_1
        # reward = -1.5*distance - np.log(distance + 1e-6)
        
        # reward of 0518 v1-run16
        reward = -2.5*distance - 0.5*np.log(distance + 1e-6)
        return reward
    def get_stop_distance(self,stop_distance):
        self.stop_distance=stop_distance
        
    def _get_terminal(self, current_info):
        # print('stop_distance={}'.format(self.stop_distance))
        return not current_info["distance"] >= self.stop_distance # 0.05

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip
