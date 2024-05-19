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
def train(num_iterations, agent, env,  evaluate, validate_steps, output, max_episode_length=None, debug=False,
          save_pic=True, env_name='ParallelCableDrivenRobot-v0', 
          change_object_point=False, object_point_pos_array=None,stop_distance=0.05):
    # num_iterations:训练阶段总step数 max_episode_length:各episode的最大step数
    # validate_steps:训练过程中, 每隔validate_steps进行一次evaluate
    output = get_output_folder(output,env_name)
    agent.is_training = True
    step = episode = episode_steps = success_episode = 0
    episode_reward = 0.
    observation = None

    while step < num_iterations:
        # reset if it is the start of episode episode的初始时刻observation=None
        if observation is None:
            
            # observation = deepcopy(env.reset())
            # agent.reset(observation)
            # # print(f'observation:{observation} shape of observation:{observation.shape}')
            # if change_object_point:
            #     index = np.random.randint(0,num_of_sample)
            #     object_point = object_point_pos_array[index,:]
            #     print('object_point={}'.format(object_point))
            #     env.sim.model.body_pos[env.sim.model.body_name2id('object_point')] = object_point
            #     env.sim.forward()
            if change_object_point:
                env.reset()
                index = np.random.randint(0, object_point_pos_array.shape[0])
                object_point = object_point_pos_array[index,:]
                env.sim.model.body_pos[env.sim.model.body_name2id('object_point')] = object_point
                env.get_stop_distance(stop_distance)
                env.sim.forward()
                # print('env.stop_distance={}'.format(stop_distance))
                print('pos of object point={}'.format(env.sim.data.get_site_xpos('object_point')))
                observation = deepcopy(env.get_observation())
                agent.reset(observation)
            else:
                observation = deepcopy(env.reset())
                agent.reset(observation)
            
        # agent pick action ...
        # step<=args.warmup时，不训练，随机生成动作；step>args.warmup时，通过actor网络产生动作，并进行训练
        if step <= args.warmup:
            action = agent.random_action()     
        else:
            action = agent.select_action(observation)
        
        action = env.action_scaling(action) # 将action等比例恢复为环境的action_space对应区间 
        # print(f'a_t={action}')
        # env response with next_observation, reward, terminate_info
        observation2, reward, done, info = env.step(action)
        observation2 = deepcopy(observation2)
        
        # 促进自然训练结束
        if done and episode_steps<max_episode_length-1:
            reward += -np.log(0.05)*(max_episode_length-episode_steps)
            success_episode += 1
        # 达到max_episode_length, 将done强制置为true
        if max_episode_length and episode_steps >= max_episode_length -1:
            done = True

        # agent observe and update policy
        agent.observe(reward, observation2, done)
        if step > args.warmup :
            agent.update_policy()
        
        # [optional] evaluate
        if evaluate is not None and validate_steps > 0 and step % validate_steps == 0:
            policy = lambda x: agent.select_action(x, decay_epsilon=False)
            validate_reward = evaluate(env, policy, debug=False, visualize=False, 
                                       object_point_pos_array=object_point_pos_array,change_object_point=change_object_point) # 此处调用__call__()方法  
            if save_pic:
                evaluate.save_results('{}/validate_reward'.format(output))
            if debug: prYellow('[Evaluate] Step_{:07d}: mean_reward:{}'.format(step, validate_reward))
            # prYellow('[Evaluate] Step_{:07d}: mean_reward:{}'.format(step, validate_reward)) # 不进入调试模式也直接打印训练结果 
            
        # [optional] save intermideate model
        if step % int(num_iterations/3) == 0:
            agent.save_model(output)

        # update 
        step += 1
        episode_steps += 1
        episode_reward += reward
        observation = deepcopy(observation2)

        if done: # end of episode
            if debug: prGreen('#{}: object_point:{} episode_reward:{} steps:{} episode_steps:{}'.format(episode,env.sim.data.get_site_xpos('object_point'),episode_reward,step,episode_steps))
            # 存入记忆回放池memory的四元组为当前状态s_t, 更新后的actor网络产生的动作, 当前奖励r_t, 完成标志done
            agent.memory.append(
                observation,
                agent.select_action(observation),
                0., False
            )

            # reset
            observation = None
            episode_steps = 0
            episode_reward = 0.
            episode += 1
    
    if debug: prPurple('n_episode:{} success_episode:{} success_rate:{}'.format(episode+1,success_episode,success_episode/(episode+1)))

def get_object_point_pos_array(data_path,num_of_sample,seed):
    # 读取flexible_point_end文件夹中的csv文件
    data_path=f"{data_path}"
    data_frame=pd.read_csv(data_path,header=0,index_col=0)    
    object_point_pos_array=data_frame.sample(n=num_of_sample,random_state=seed).iloc[:,-3:].to_numpy() # 初始末端点位置
    prLightPurple('shape of object point pos array:{}'.format(object_point_pos_array.shape))
    return object_point_pos_array

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch on TORCS with Multi-modal')

    parser.add_argument('--mode', default='train', type=str, help='support option: train/test')
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
    parser.add_argument('--validate_episodes', default=20, type=int, help='how many episode to perform during validate experiment')
    # parser.add_argument('--max_episode_length', default=500, type=int, help='')
    parser.add_argument('--validate_steps', default=2000, type=int, help='how many steps to perform a validate experiment')
    parser.add_argument('--output', default='D:\\Research\\ICCSSE\\ParalllelCableDrivenRobot\\algorithm\\pytorch-ddpg-master\\output', type=str, help='')
    # parser.add_argument('--debug', dest='debug', action='store_true') # action='store_true'：命令行调用参数--debug但不赋值时，自动赋值True
    parser.add_argument('--init_w', default=0.003, type=float, help='') 
    parser.add_argument('--train_iter', default=500000, type=int, help='train iters each timestep')
    parser.add_argument('--epsilon', default=50000, type=int, help='linear decay of exploration policy')
    parser.add_argument('--seed', default=123456, type=int, help='')
    parser.add_argument('--resume', default='default', type=str, help='Resuming model path for testing')
    # parser.add_argument('--l2norm', default=0.01, type=float, help='l2 weight decay') # TODO
    # parser.add_argument('--cuda', dest='cuda', action='store_true') # TODO
    parser.add_argument('--debug', default=True, type=bool, help='')
    parser.add_argument('--data_path',default='',type=str,help='sample points from work space')
    parser.add_argument('--num_of_sample',default=500,type=int,help='num of sampled points from work space')
    parser.add_argument('--change_object_point',default=False,type=bool,help='whether change pos of object point in the training stage')
    parser.add_argument('--stop_distance',default=0.05,type=float,help='when distance=stop_distance, done is True')
    
    args = parser.parse_args()
    # args.output = get_output_folder(args.output, args.env)
    print(f'env:{args.env}')
    # print(f'args.output:{args.output}')
    

    # env = NormalizedEnv(gym.make(args.env)) # 对环境进行归一化处理
    env = gym.make(args.env)
    # env.stop_distance = args.stop_distance
    nb_states = env.observation_space.shape[0]
    nb_actions = env.action_space.shape[0]
    print(f'env={env}')
    print(f'nb_states={nb_states} nb_actions={nb_actions}')
    print(f'max_episode_steps={env._max_episode_steps}')
    # print(f'stop_distance={env.stop_distance}')
    # print(f'init_qpos={env.init_qpos} shape_of_init_qpos={env.init_qpos.shape}')
    # print('pos of point_end={}'.format(env.sim.data.get_site_xpos('point_end')))
    
    if args.seed > 0:
        np.random.seed(args.seed)
        env.seed(args.seed)
        env.action_space.seed(args.seed)
    
    # 判断是否改变目标点
    if args.change_object_point:
        object_point_pos_array = get_object_point_pos_array(args.data_path, args.num_of_sample, args.seed)
    else:
        object_point_pos_array = None
    
    agent = DDPG(nb_states, nb_actions, args)
    evaluate = Evaluator(args.validate_episodes, 
        args.validate_steps, max_episode_length=env._max_episode_steps,stop_distance=args.stop_distance)

    if args.mode == 'train':
        train(args.train_iter, agent, env, evaluate, args.validate_steps, args.output, 
            max_episode_length=env._max_episode_steps, debug=args.debug, env_name=args.env,
            change_object_point=args.change_object_point, object_point_pos_array=object_point_pos_array,
            stop_distance=args.stop_distance)

    else:
        raise RuntimeError('undefined mode {}'.format(args.mode))
