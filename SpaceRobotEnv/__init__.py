# import logging
from gym.envs.registration import register

# logger = logging.getLogger(__name__)

register(
    id="ParallelCableDrivenRobot-v0",
    entry_point="SpaceRobotEnv.envs:ParallelCableDrivenRobot_v0",
    max_episode_steps=250,
)

register(
    id="ParallelCableDrivenRobot-v1",
    entry_point="SpaceRobotEnv.envs:ParallelCableDrivenRobot_v1",
    max_episode_steps=250,
)

register(
    id="ParallelCableDrivenRobot-v11",
    entry_point="SpaceRobotEnv.envs:ParallelCableDrivenRobot_v11",
    max_episode_steps=250,
)

register(
    id="ParallelCableDrivenRobot-v2",
    entry_point="SpaceRobotEnv.envs:ParallelCableDrivenRobot_v2",
    max_episode_steps=250,
)

register(
    id="ParallelCableDrivenRobot-v21",
    entry_point="SpaceRobotEnv.envs:ParallelCableDrivenRobot_v21",
    max_episode_steps=250,
)