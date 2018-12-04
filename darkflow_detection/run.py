import updateddark import darkflow_prediction
from danger_avoidance import make_rl

pred = darkflow_prediction()
env = make_rl()
lstm_model = load_model("snippet.mp4.h5")

step_limit = 2000
step_num = 0

while step_num < step_limit: 
    locations = lstm_model.predict() #TODO: takes in current locations of cars
    obs, reward, done, info = env.step(action, locations)
    print("No crash! Observation taken:", obs)
    #TODO: add some sort of visual drawing move?
    step_num += 1

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


