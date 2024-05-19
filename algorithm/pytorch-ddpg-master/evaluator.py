
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat

from util import *

class Evaluator(object):

    def __init__(self, num_episodes, interval, max_episode_length=None, stop_distance=0.05):
        self.num_episodes = num_episodes
        self.max_episode_length = max_episode_length
        self.interval = interval
        self.results = np.array([]).reshape(num_episodes,0)
        self.stop_distance = stop_distance

    def __call__(self, env, policy, debug=False, visualize=False,
                 object_point_pos_array=None, change_object_point=False): # 在对象实例后面加上括号并传入参数时，Python 会自动调用该对象的 __call__ 方法

        self.is_training = False
        self.object_point_pos_array = object_point_pos_array
        observation = None
        result = []
        success_episode = 0

        for episode in range(self.num_episodes):

            # reset at the start of episode
            if change_object_point:
                observation = self.reset_of_change(env=env)
            else:
                observation = env.reset()
            episode_steps = 0
            episode_reward = 0.
                
            assert observation is not None

            # start episode
            done = False
            while not done:
                # basic operation, action ,reward, blablabla ...
                action = policy(observation)
                action = env.action_scaling(action) #将action等比例恢复为环境的action_space对应区间
                # print(f'a_t={action}')
                
                observation, reward, done, info = env.step(action)
                if done and episode_steps<self.max_episode_length-1:
                    reward+=5*(self.max_episode_length-episode_steps)
                    success_episode += 1
                if self.max_episode_length and episode_steps >= self.max_episode_length -1:
                    done = True
                
                if visualize:
                    # env.render(mode='human')
                    env.render()

                # update
                episode_reward += reward
                episode_steps += 1
            # prYellow 将输入的内容以黄色打印到终端
            if debug: prYellow('[Evaluate] #Episode{}: object_point:{} episode_reward:{} episode_steps:{}'.format(episode,env.sim.data.get_site_xpos('object_point'),episode_reward,episode_steps))
            result.append(episode_reward)

        result = np.array(result).reshape(-1,1)
        self.results = np.hstack([self.results, result])
        if debug: prPurple('[Evaluate] success_rate:{}'.format(success_episode/self.num_episodes))
        return np.mean(result)

    def save_results(self, fn):

        y = np.mean(self.results, axis=0)
        error=np.std(self.results, axis=0)
                    
        x = range(0,self.results.shape[1]*self.interval,self.interval)
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        plt.xlabel('Timestep')
        plt.ylabel('Average Reward')
        ax.errorbar(x, y, yerr=error, fmt='-o')
        plt.savefig(fn+'.png')
        savemat(fn+'.mat', {'reward':self.results})
        plt.close()
        
    def reset_of_change(self, env):
        env.reset()
        index = np.random.randint(0,self.object_point_pos_array.shape[0])
        object_point = self.object_point_pos_array[index,:]
        env.sim.model.body_pos[env.sim.model.body_name2id('object_point')] = object_point
        env.get_stop_distance(self.stop_distance)
        # print('env.stop_distance={}'.format(self.stop_distance))
        env.sim.forward()
        obs = env.get_observation()
        
        return obs