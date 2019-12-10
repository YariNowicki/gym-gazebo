from gym.envs.registration import register

register(
    id='gazebo-v0',
    entry_point='gym_gazebo.envs:GazeboEnv',
)

register(
    id='hector-v0',
    entry_point='gym_gazebo.envs.hector:GazeboWorldHectorLaserCamera'
)
