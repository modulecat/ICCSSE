#!/usr/bin/env python3 

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

# gym.undo_logger_setup()

# print的颜色规则: 黄色为evaluate和test 绿色为train

def test(num_episodes, agent, env, evaluate, model_path, visualize=True, debug=False,
         object_point_pos_array=None,change_object_point=False):

    agent.load_weights(model_path)
    agent.is_training = False
    agent.eval()
    policy = lambda x: agent.select_action(x, decay_epsilon=False)

    # for i in range(num_episodes):        
        # validate_reward = evaluate(env, policy, debug=debug, visualize=visualize)
        # if debug: prYellow('[Evaluate] #{}: mean_reward:{}'.format(i, validate_reward))

    validate_reward = evaluate(env, policy, debug=debug, visualize=visualize,
                               object_point_pos_array=object_point_pos_array, change_object_point=change_object_point)
    if debug: prYellow('[Evaluate] validate_episodes:{} mean_reward:{}'.format(num_episodes, validate_reward))

def get_object_point_pos_array(data_path,num_of_sample,seed):
    # 读取flexible_point_end文件夹中的csv文件
    data_path=f"{data_path}"
    data_frame=pd.read_csv(data_path,header=0,index_col=0)    
    object_point_pos_array=data_frame.sample(n=num_of_sample,random_state=seed).iloc[:,-3:].to_numpy() # 初始末端点位置
    prLightPurple('shape of object point pos array:{}'.format(object_point_pos_array.shape))
    return object_point_pos_array

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch on TORCS with Multi-modal')

    parser.add_argument('--mode', default='test', type=str, help='support option: train/test')
    parser.add_argument('--env', default='ParallelCableDrivenRobot-v0', type=str, help='open-ai gym environment')
    parser.add_argument('--hidden1', default=400, type=int, help='hidden num of first fully connect layer')
    parser.add_argument('--hidden2', default=300, type=int, help='hidden num of second fully connect layer')
    parser.add_argument('--rate', default=0.001, type=float, help='learning rate')
    parser.add_argument('--prate', default=0.0001, type=float, help='policy net learning rate (only for DDPG)')
    parser.add_argument('--warmup', default=100, type=int, help='time without training but only filling the replay memory')
    parser.add_argument('--discount', default=0.99, type=float, help='')
    parser.add_argument('--bsize', default=64, type=int, help='minibatch size')
    parser.add_argument('--rmsize', default=6000000, type=int, help='memory size')
    parser.add_argument('--window_length', default=1, type=int, help='')
    parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
    parser.add_argument('--ou_theta', default=0.15, type=float, help='noise theta')
    parser.add_argument('--ou_sigma', default=0.2, type=float, help='noise sigma') 
    parser.add_argument('--ou_mu', default=0.0, type=float, help='noise mu') 
    parser.add_argument('--validate_episodes', default=100, type=int, help='how many episode to perform during validate experiment')
    # parser.add_argument('--max_episode_length', default=500, type=int, help='')
    parser.add_argument('--validate_steps', default=2000, type=int, help='how many steps to perform a validate experiment')
    parser.add_argument('--output', default='D:\\Research\\ICCSSE\\ParalllelCableDrivenRobot\\algorithm\\pytorch-ddpg-master\\output', type=str, help='')
    # parser.add_argument('--debug', dest='debug', action='store_true') # action='store_true'：命令行调用参数--debug但不赋值时，自动赋值True
    parser.add_argument('--init_w', default=0.003, type=float, help='') 
    parser.add_argument('--train_iter', default=500000, type=int, help='train iters each timestep')
    parser.add_argument('--epsilon', default=50000, type=int, help='linear decay of exploration policy')
    parser.add_argument('--seed', default=1234, type=int, help='')
    parser.add_argument('--resume', default='default', type=str, help='Resuming model path for testing')
    # parser.add_argument('--l2norm', default=0.01, type=float, help='l2 weight decay') # TODO
    # parser.add_argument('--cuda', dest='cuda', action='store_true') # TODO
    parser.add_argument('--debug', default=True, type=bool, help='')
    parser.add_argument('--data_path',default='',type=str,help='sample points from work space')
    parser.add_argument('--change_object_point',default=False,type=bool,help='whether change pos of object point in the training stage')
    parser.add_argument('--stop_distance',default=0.05,type=float,help='when distance=stop_distance, done is True')
    
    args = parser.parse_args()
    print(f'env:{args.env}')

    # env = NormalizedEnv(gym.make(args.env)) # 对环境进行归一化处理
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

    # 将遍历得到的初始末端点定义为训练的目标点
    if args.change_object_point:
        object_point_pos_array=get_object_point_pos_array(args.data_path,args.validate_episodes,args.seed)
    else:
        object_point_pos_array=None
    
    agent = DDPG(nb_states, nb_actions, args)
    evaluate = Evaluator(args.validate_episodes, 
        args.validate_steps, max_episode_length=env._max_episode_steps, stop_distance=args.stop_distance)

    if args.mode == 'test':
        test(args.validate_episodes, agent, env, evaluate, args.resume,
            visualize=True, debug=args.debug,
            object_point_pos_array=object_point_pos_array, change_object_point=args.change_object_point)

    else:
        raise RuntimeError('undefined mode {}'.format(args.mode))
