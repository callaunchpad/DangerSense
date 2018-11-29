import gym
from gym import error, spaces, utils
from gym.utils import seeding

class Env(gym.Env):
	metadata = {'render.modes': ['human']}

	def __init__(self):
		self.observation_space = spaces.Discrete(3)
		self.action_space = spaces.Discrete(3)
		self.num_steps = 0

	def step(self, action):
		self._take_action(action)
		self.status = self.env.step()
		reward = self._get_reward()
		ob = self.env.getState()
		self.num_steps += 1
		episode_over = (self.num_steps >= 20)
		return ob, reward, episode_over, {}

	def reset(self):
		pass

	def render(self, mode='human', close=False):
		pass

	def _take_action(self, action):
		pass

	def _get_reward(self):
	    """ Reward is given for XY. """
	    if self.status == 0:
	        return 1
	    elif self.status == 1:
	        return self.somestate ** 2
	    else:
	        return 0
