from gym.envs.registration import register

register(
    id='hector-v0',
    entry_point='gym_hector.envs:HectorEnv',
)

register(
    id='hector-extrahard-v0',
    entry_point='gym_hector.envs:HectorExtraHardEnv',
)