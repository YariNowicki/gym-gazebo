from gym.envs.registration import register

register(
    id='gazebo-v0',
    entry_point='gym_hector.envs:GazeboEnv',
)
