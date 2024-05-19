import argparse
import numpy as np
import torch
from sac import SAC

import sys
sys.path.append(fr"D:\Research\ICCSSE\ParalllelCableDrivenRobot")
import SpaceRobotEnv
import gym

parser = argparse.ArgumentParser(description='Soft Actor-Critic Args')
parser.add_argument('--env_name', default="SingleArmFixedBaseFixedTarget-v0")
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--model_path', type=str, help='path of loaded model')
args = parser.parse_args()

# Environment
# env = NormalizedActions(gym.make(args.env_name))
env = gym.make(args.env_name)
env.seed(args.seed)
env.action_space.seed(args.seed)

torch.manual_seed(args.seed)
np.random.seed(args.seed)


model_path = args.model_path


# Agent
agent = torch.load(model_path)


avg_reward = 0.
episodes = 100
for _  in range(episodes):
    state = env.reset()
    episode_reward = 0
    done = False
    env.render()
    while not done:
        action = agent.select_action(state, evaluate=True)

        next_state, reward, done, info = env.step(action)
        env.render()
        episode_reward += reward
        state = next_state
        # print(info)
    # print("=" * 100)
    avg_reward += episode_reward
    print(episode_reward)
avg_reward /= episodes



print("----------------------------------------")
print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
print("----------------------------------------")


env.close()

