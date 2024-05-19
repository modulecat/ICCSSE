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

# gym.undo_logger_setup()

# print的颜色规则: 黄色为evaluate和test 绿色为train
def train(num_iterations, agent, env,  evaluate, validate_steps, output, max_episode_length=None, debug=False,save_pic=True, env_name='ParallelCableDrivenRobot-v0'):
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
            observation = deepcopy(env.reset())
            agent.reset(observation)
            # print(f'observation:{observation} shape of observation:{observation.shape}')
            
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
            validate_reward = evaluate(env, policy, debug=False, visualize=False) # 此处调用__call__()方法  
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
            if debug: prGreen('#{}: episode_reward:{} steps:{} episode_steps:{}'.format(episode,episode_reward,step,episode_steps))
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

def test(num_episodes, agent, env, evaluate, model_path, visualize=True, debug=False):

    agent.load_weights(model_path)
    agent.is_training = False
    agent.eval()
    policy = lambda x: agent.select_action(x, decay_epsilon=False)

    for i in range(num_episodes):        
        validate_reward = evaluate(env, policy, debug=debug, visualize=visualize)
        if debug: prYellow('[Evaluate] #{}: mean_reward:{}'.format(i, validate_reward))


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
    parser.add_argument('--seed', default=-1, type=int, help='')
    parser.add_argument('--resume', default='default', type=str, help='Resuming model path for testing')
    # parser.add_argument('--l2norm', default=0.01, type=float, help='l2 weight decay') # TODO
    # parser.add_argument('--cuda', dest='cuda', action='store_true') # TODO
    parser.add_argument('--debug', default=True, type=bool, help='')
    
    args = parser.parse_args()
    # args.output = get_output_folder(args.output, args.env)
    print(f'env:{args.env}')
    # print(f'args.output:{args.output}')
    
    
    # if args.resume == 'default':
    #     args.resume = 'D:\\Research\\ICCSSE\\ParalllelCableDrivenRobot\\algorithm\\pytorch-ddpg-master\\output\\{}-run1'.format(args.env)

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
    
    
    # env.reset()
    # print(f'init_qpos={env.init_qpos}')
    # env.init_qpos[-1]=20/180*np.pi
    # env.set_state(env.init_qpos,env.init_qvel)
    # env.render()
    # print('pos of point_end={}'.format(env.sim.data.get_site_xpos('point_end')))
    
    # action=env.action_space.sample()
    # print(f'action={action}')
    
    
    agent = DDPG(nb_states, nb_actions, args)
    evaluate = Evaluator(args.validate_episodes, 
        args.validate_steps, max_episode_length=env._max_episode_steps)

    if args.mode == 'train':
        train(args.train_iter, agent, env, evaluate, 
            args.validate_steps, args.output, max_episode_length=env._max_episode_steps, debug=args.debug, env_name=args.env)

    elif args.mode == 'test':
        test(args.validate_episodes, agent, env, evaluate, args.resume,
            visualize=True, debug=args.debug)

    else:
        raise RuntimeError('undefined mode {}'.format(args.mode))
