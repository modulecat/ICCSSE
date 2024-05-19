import numpy as np
import argparse
from copy import deepcopy
import torch
import gym
import sys
sys.path.append(fr"D:\Research\ICCSSE\ParalllelCableDrivenRobot")
import SpaceRobotEnv

from normalized_env import NormalizedEnv
from evaluator import Evaluator
from ddpg import DDPG
from util import *
import pandas as pd
import os

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch on TORCS with Multi-modal')
    parser.add_argument('--env', default='ParallelCableDrivenRobot-v1', type=str, help='open-ai gym environment')
    parser.add_argument('--seed', default=123456, type=int, help='')
    args = parser.parse_args()
    
    env = gym.make(args.env)

    if args.seed > 0:
        np.random.seed(args.seed)
        env.seed(args.seed)

    nb_states = env.observation_space.shape[0]
    nb_actions = env.action_space.shape[0]
    print(f'nb_states={nb_states} nb_actions={nb_actions}')
    print(f'max_episode_steps={env._max_episode_steps}')
    
    # init_qpos为29元素一维numpy数组,从0~28依次为自由关节位姿7,第一臂段平移关节4,第一臂段旋转关节2，第二臂段平移关节4,第二臂段旋转关节2,第三臂段平移关节8,第三臂段旋转关节2
    print(f'init_qpos={env.init_qpos} shape_of_init_qpos={env.init_qpos.shape}')
    # print('pos of point_end={}'.format(env.sim.data.get_site_xpos('point_end')))
    
    # 给定关节角度, 获取末端位置
    # v0
    # env.reset()
    # print(f'init_qpos={env.init_qpos}')
    # env.init_qpos[-1]=-5/180*np.pi
    # env.init_qpos[-11]=-5/180*np.pi
    # env.init_qpos[-17]=-5/180*np.pi
    # env.set_state(env.init_qpos,env.init_qvel)
    # print('pos of point_end={}'.format(env.sim.data.get_site_xpos('point_end')))
    # while 1:
        # env.render()
    
    # v1
    data_dict={'joint1_1':[],'joint1_2':[],'joint2_1':[],'joint2_2':[],'joint3_1':[],'joint3_2':[],'point_endx':[],'point_endy':[],'point_endz':[]}
    
    # for flag1_1 in np.linspace(-1,1,5):
    #     for flag1_2 in np.linspace(-1,1,5):
    #         for flag2_1 in np.linspace(-1,1,5):
    #             for flag2_2 in np.linspace(-1,1,5):
    #                 for flag3_1 in np.linspace(-1,1,5):
    #                     for flag3_2 in np.linspace(-1,1,5):
    #                         Flag = np.array([flag1_1, flag1_2, flag2_1, flag2_2, flag3_1, flag3_2])
    #                         Joint_angle = Flag*(5/180*np.pi)
    #                         env.reset()
    #                         env.init_qpos[-1] = Joint_angle[-1]
    #                         env.init_qpos[-2] = Joint_angle[-2]
    #                         env.init_qpos[-7] = Joint_angle[-3]
    #                         env.init_qpos[-8] = Joint_angle[-4]
    #                         env.init_qpos[-13] = Joint_angle[-5]
    #                         env.init_qpos[-14] = Joint_angle[-6]
    #                         env.set_state(env.init_qpos,env.init_qvel)
    #                         point_end=env.sim.data.get_site_xpos('point_end')
    #                         data_array=np.concatenate((Joint_angle, point_end))
    #                         index=0
    #                         for key in data_dict:
    #                             data_dict[key].append(data_array[index])
    #                             index+=1
                                
    # 第一象限工作空间                    
    # for flag1_1 in np.linspace(0,1,5):
    #     for flag1_2 in np.linspace(0,1,5):
    #         for flag2_1 in np.linspace(0,1,5):
    #             for flag2_2 in np.linspace(0,1,5):
    #                 for flag3_1 in np.linspace(0,1,5):
    #                     for flag3_2 in np.linspace(0,1,5):
    #                         Flag = np.array([flag1_1, flag1_2, flag2_1, flag2_2, flag3_1, flag3_2])
    #                         Joint_angle = Flag*(5/180*np.pi)
    #                         env.reset()
    #                         env.init_qpos[-1] = Joint_angle[-1]
    #                         env.init_qpos[-2] = Joint_angle[-2]
    #                         env.init_qpos[-7] = Joint_angle[-3]
    #                         env.init_qpos[-8] = Joint_angle[-4]
    #                         env.init_qpos[-13] = Joint_angle[-5]
    #                         env.init_qpos[-14] = Joint_angle[-6]
    #                         env.set_state(env.init_qpos,env.init_qvel)
    #                         point_end=env.sim.data.get_site_xpos('point_end')
    #                         data_array=np.concatenate((Joint_angle, point_end))
    #                         index=0
    #                         for key in data_dict:
    #                             data_dict[key].append(data_array[index])
    #                             index+=1
                                
    
    # 缩小角度范围[-3deg, 3deg]                           
    for flag1_1 in np.linspace(-1,1,3):
        for flag1_2 in np.linspace(-1,1,3):
            for flag2_1 in np.linspace(-1,1,3):
                for flag2_2 in np.linspace(-1,1,3):
                    for flag3_1 in np.linspace(-1,1,3):
                        for flag3_2 in np.linspace(-1,1,3):
                            Flag = np.array([flag1_1, flag1_2, flag2_1, flag2_2, flag3_1, flag3_2])
                            Joint_angle = Flag*(3/180*np.pi)
                            env.reset()
                            env.init_qpos[-1] = Joint_angle[-1]
                            env.init_qpos[-2] = Joint_angle[-2]
                            env.init_qpos[-7] = Joint_angle[-3]
                            env.init_qpos[-8] = Joint_angle[-4]
                            env.init_qpos[-13] = Joint_angle[-5]
                            env.init_qpos[-14] = Joint_angle[-6]
                            env.set_state(env.init_qpos,env.init_qvel)
                            point_end=env.sim.data.get_site_xpos('point_end')
                            data_array=np.concatenate((Joint_angle, point_end))
                            index=0
                            for key in data_dict:
                                data_dict[key].append(data_array[index])
                                index+=1
    # for key in data_dict:
        # print('data_dict[{}]:{},shape of data_dict[{}]:{}'.format(key,data_dict[key],key,len(data_dict[key])))
    
    data_frame = pd.DataFrame(data_dict)
    data_path_prefix = os.path.join(fr'D:\Research\ICCSSE\ParalllelCableDrivenRobot\algorithm\pytorch-ddpg-master',fr'flexible_point_end')
    print('data_path_prefix:{}'.format(data_path_prefix))
    if not os.path.isdir(data_path_prefix):
        os.makedirs(data_path_prefix)
        
    # 工作空间储存文件
    # csv_name=f'{args.env}-seed-{args.seed}-trial.csv'
    
    # 工作空间储存文件[-5deg,5deg]均匀取5个点
    # csv_name=f'{args.env}-seed-{args.seed}-5.csv'
    
    # 第一象限工作空间
    # csv_name=f'{args.env}-seed-{args.seed}-5-singlequadrant.csv'
    
    # 缩小角度范围[-3deg, 3deg]
    csv_name=f'{args.env}-seed-{args.seed}-3.csv'
    
    data_path=os.path.join(data_path_prefix,csv_name)
    data_frame.to_csv(data_path)
   
    
    
    # action=env.action_space.sample()
    # print(f'action={action}')
    