import os
import cv2
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from keras import backend as K
from keras.datasets import mnist, cifar10
from keras.models import Sequential, load_model, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from keras.optimizers import RMSprop, Adam, SGD
from keras.models import clone_model, model_from_json
from keras.callbacks import EarlyStopping
import gym
import rlenv1

class testRL_LSTM():

	def __init__(self):
		# Load trained RL model from file
		json_file = open('rlmodel.json', 'r')
		model_json = json_file.read()
		json_file.close()
		self.model = model_from_json(model_json)
		self.model.load_weights("rlmodel.h5")
		self.actiondescs = {
			0: "maintain speed",
			1: "decrease speed",
			2: "increase speed",
			3: "swerve",
		}

	def compute_state(self, centroid, bounding_box):
		# Define our actual state here: 
		for location in next_locations: # list of locations of cars (x,y) in the next time frame
			x = location[0]
			y = location[1]
			# TODO: Velocity is relative based on the car in front of it? Not going to work (Ask Olivia for equations)...
			if math.abs(x - self.centerX) < 100: # naively just look at the x location being similar to the middle, compare y
				if y < self.crash_dist and self.state < 4:
					self.state = 4 # car crash
				elif y < self.swerve_dist and self.state < 3:
					self.state = 3 # swerve
				elif y < self.slow_dist and self.state < 2:
					self.state = 2 # slow
				else:
					if self.state < 1:
						self.state = 1 # maintain
			else:
				self.state = 0 # car not in lane

	def react(self, state):
		action = np.argmax(self.model.predict(np.identity(5)[state:state + 1]))
		print("Action:", self.actiondescs[action])
		return action
