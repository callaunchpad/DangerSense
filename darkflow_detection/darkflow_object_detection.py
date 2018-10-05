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
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import RMSprop, Adam, SGD
from keras.models import clone_model
from keras.callbacks import EarlyStopping
# from yolo_utils import *
from darkflow.net.build import TFNet
import re
import time

class darkflow_prediction():


	def __init__(self):
		self.options = {"model": "cfg/yolo.cfg", "load": "bin/yolov2.weights", "threshold": 0.5}
		self.tfnet = TFNet(self.options)

	def image(self, image_file):
		self.image = cv2.imread(image_file, 1)
		self.result = self.tfnet.return_predict(self.image)
		print(self.result)
		self.print_box()
		cv2.waitKey(0)

	def print_box(self):
		font = cv2.FONT_HERSHEY_PLAIN
		for i in range(len(self.result)):
			coordtl = (self.result[i]['topleft']['x'], self.result[i]['topleft']['y'])
			coordbr = (self.result[i]['bottomright']['x'], self.result[i]['bottomright']['y'])
			cv2.rectangle(self.image,coordtl,coordbr,(0,255,0),2)
			s = str(self.result[i]['label'] + ": " + str(self.result[i]['confidence']))
			text_coord = (coordtl[0], coordtl[1]-10)
			cv2.putText(self.image, s, text_coord, font, 1, (250,250,0))
		cv2.imshow("memes", self.image)

	def video(self, video_file):
		self.video = cv2.VideoCapture(video_file)
		while self.video.isOpened():
			ret, self.image = self.video.read()
			self.result = self.tfnet.return_predict(self.image)
			self.print_box()
			cv2.waitKey(1)
	
	def video_with_frame_drop(self, video_file, FPS=30):
		self.video = cv2.VideoCapture(video_file)
		skip_frames = 0
		t = time.time()
		try:
			while self.video.isOpened():
				for i in range(skip_frames):
					_, _ = self.video.read()
				ret, self.image = self.video.read()
				self.result = self.tfnet.return_predict(self.image)
				self.print_box()
				cv2.waitKey(1)
				skip_frames = int((time.time()-t)*FPS)
				t = time.time()
		except AssertionError:
			pass

pred = darkflow_prediction()
# pred.image("../cars2.jpg")
pred.video_with_frame_drop("../cars_video.mp4")
