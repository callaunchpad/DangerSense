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

class darkflow_prediction():

	def __init__(self, image_file):
		options = {"model": "cfg/yolo.cfg", "load": "bin/yolov2.weights", "threshold": 0.5}
		self.tfnet = TFNet(options)
		self.image = cv2.imread(image_file, 1)
		self.result = self.tfnet.return_predict(self.image)
		print(self.result)
		self.print_box()

	def print_box(self):
		for i in range(len(self.result)):
			coordtl = (self.result[i]['topleft']['x'], self.result[i]['topleft']['y'])
			coordbr = (self.result[i]['bottomright']['x'], self.result[i]['bottomright']['y'])
			cv2.rectangle(self.image,coordtl,coordbr,(0,255,0),2)
		cv2.imshow("memes", self.image)
    
	def video(self, video_file):
		cap = cv2.VideoCapture(video_file)
		while cap.isOpened():
			ret, self.image = cap.read()
			self.result = self.tfnet.return_predict(self.image)
			self.print_box()
			cv2.waitKey(1)

pred = darkflow_prediction("../cars3.jpg")
pred.video('../car1min.avi')
