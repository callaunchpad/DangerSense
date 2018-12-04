import gym
from gym import error, spaces, utils
from gym.utils import seeding
from random import randint

class Env(gym.Env):
	metadata = {'render.modes': ['human']}

	# States
	# 	State 0 = car is not in your lane
	# 	State 1 = car is in your lane, but not close
	# 	State 2 = car is in your lane and close
	#	State 3 = car is in your lane and super close
	#	State 4 = crash
	# Actions
	# 	Action 0 = maintain speed
	# 	Action 1 = decrease speed
	# 	Action 2 = increase speed
	# 	Action 3 = swerve

	def __init__(self, num_states=5, num_actions=4, put_in_danger=0.7):
		self.num_states = num_states
		self.num_actions = num_actions
		self.put_in_danger = put_in_danger
		self.state = randint(0, 1) #0
		self.action_space = spaces.Discrete(self.num_actions)
		self.observation_space = spaces.Discrete(self.num_states)
		self.seed()
		self.crash_state = num_states - 1
		self.action0changes = { # maintain speed
			0: [1, 0],
			1: [0.5, 1],
			2: [-0.4, 3],
			3: [-0.7, 4],
			4: [0, 0],
		}
		self.action1changes = { # decrease speed
			0: [0.1, 0],
			1: [0.5, 1],
			2: [1, 1],
			3: [0.4, 2],
			4: [0, 0],
		}
		self.action2changes = { # increase speed
			0: [0.3, 0],
			1: [0.1, 2],
			2: [-0.8, 4],
			3: [-1, 4],
			4: [0, 0],
		}
		self.action3changes = { # swerve
			0: [-0.1, 1],
			1: [0, 0],
			2: [0.4, 0],
			3: [1, 0],
			4: [0, 0],
		}

	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def step(self, action, count):
		assert self.action_space.contains(action)
		prevstate = self.state
		if action == 0:
			reward, self.state = self.action0changes[self.state]
		if action == 1:
			reward, self.state = self.action1changes[self.state]
		if action == 2:
			reward, self.state = self.action2changes[self.state]
		if action == 3:
			reward, self.state = self.action3changes[self.state]
		if self.np_random.rand() < self.put_in_danger and self.state != self.crash_state:
			self.state = randint(2, 3)
		done = (count == 2000)
		return self.state, reward, done, {}

	def reset(self):
		self.state = randint(0, 1) #0
		return self.state

	def render(self, mode='human', close=False):
		pass
