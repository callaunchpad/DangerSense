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
			3: "swerve"}

		# Load trained LSTM model from file 
		self.lstm_model = load_model("snippet.mp4.h5")

		self.centerX = 1000 # TODO: Should be passed in by the video
		self.crash_area = 30
		self.swerve_area = 60
		self.slow_area = 90
		self.mantain_area = 120

	def compute_state(self, centroid, bounding_box):
		# Define our actual state here: 
		centroid = self.model.predict(centroid)
		new_box = self.lstm_model.predict(bounding_box)
		area = new_box[0] * new_box[1]
		x = centroid[0]
		
		if math.abs(x - self.centerX) < 100: # naively just look at the x location being similar to the middle, compare area for closeness
			if area > self.crash_area: 
				self.state = 4 # car crash
			elif area > self.swerve_area: 
				self.state = 3 # swerve
			elif area < self.slow_area: 
				self.state = 2 # slow
			else: 
				self.state = 1 # maintain
		else: 
			self.state = 0 # car not in lane

	def react(self, state):
		action = np.argmax(self.model.predict(np.identity(5)[state:state + 1]))
		print("Action:", self.actiondescs[action])
		return action
