from gym.envs.registration import registry, register, make, spec

register(
    id='rlenv1-v0',
    entry_point='rlenv1.envs:Env',
)
# register(
#     id='rlenv1-extrahard-v0',
#     entry_point='rlenv1.envs:ExtraHardEnv',
# )