import os
import cv2
import time
import argparse
import multiprocessing
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import keras
from keras import backend as K
from keras.datasets import mnist, cifar10
from keras.models import Sequential, load_model, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LSTM
from keras.optimizers import RMSprop, Adam, SGD
from keras.models import clone_model
from keras.callbacks import EarlyStopping
from darkflow.net.build import TFNet
import re
import time
from sklearn.cluster import DBSCAN
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pickle

def getXYAll(object_trajectories):
	totalData = []
	totalOutput = []
	for obj in object_trajectories:
		objectData = []
		objectOutput = []
		for i in range(0, len(object_trajectories[obj])-5):
			data = []
			for j in range(i, i+5):
				data.append([object_trajectories[obj][j]["x"], object_trajectories[obj][j]["y"]])
			objectOutput.append([object_trajectories[obj][i+5]["x"], object_trajectories[obj][i+5]["y"]]) # next position after 5 frames
			objectData.append(data)
		totalData.append(objectData)
		totalOutput.append(objectOutput)
	# append single sample
	# print('objectData', objectData) # [[[x, y], [x, y], [x, y], [x, y], [x, y]], [[x, y], [x, y], [x, y], [x, y], [x, y]]]
									# 1 sample, 5 time steps, 2 features
	# print('objectOutput', objectOutput) # output matches samples [[x, y], [x, y], [x, y], [x, y], [x, y]]
	return np.array(totalData[0]), np.array(totalOutput[0])

def predict():
	with open('object_trajectories.pickle', 'rb') as handle:
		object_trajectories = pickle.load(handle)

	dataIn, dataOut = getXYAll(object_trajectories)
	print(dataIn.shape)
	print(dataOut.shape)
	model = Sequential()
	model.add(LSTM(200, input_shape=(5, 2)))
	model.add(Dense(2))
	model.compile(loss='mean_squared_error', optimizer='adam')
	model.fit(dataIn, dataOut, epochs=100, batch_size=1, verbose=2)

	trainPredict = model.predict(dataIn)

	cap = cv2.VideoCapture("../snippet3.mp4")
	cap.set(1, cap.get(7)-50)
	ret, img = cap.read()
	cap.release()

	plt.scatter(dataOut[:,0], dataOut[:,1])
	plt.imshow(img)
	plt.show()
	plt.scatter(trainPredict[:,0], trainPredict[:,1])
	plt.imshow(img)
	plt.show()
