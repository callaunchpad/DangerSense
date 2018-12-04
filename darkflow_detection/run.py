import updateddark import darkflow_prediction
from danger_avoidance import make_rl

pred = darkflow_prediction()
env = make_rl()



obs, reward, done, info = env.step(action)

# States
# 	State 0 = car is not in your lane
# 	State 1 = car is in your lane, but not close
# 	State 2 = car is in your lane and close
#	State 3 = crash
# Actions
# 	Action 0 = maintain speed
# 	Action 1 = decrease speed
# 	Action 2 = increase speed
# 	Action 3 = swerve
