from gym.envs.registration import register

register(
    id='rlenv1-v0',
    entry_point='rlenv1.envs:Env',
)
