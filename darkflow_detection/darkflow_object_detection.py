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
		options = {"model": "cfg/yolo.cfg", "load": "bin/yolov2.weights", "threshold": 0.1}
		tfnet = TFNet(options)
		self.image = cv2.imread(image_file, 1)
		self.result = tfnet.return_predict(self.image)
		self.print_box()

	def print_box(self):
		fig, axis = plt.subplots(1)
		axis.imshow(self.image)
		for i in range(len(self.result)):
			x_start = self.result[i]['topleft']['x']
			y_start = self.result[i]['topleft']['y']
			width = self.result[i]['bottomright']['x'] - x_start
			height = self.result[i]['bottomright']['y'] - y_start
			rect = patches.Rectangle((x_start, y_start), width, height, linewidth=1, edgecolor='r', facecolor='none')
			axis.add_patch(rect)
		plt.show()

pred = darkflow_prediction("../cars1.png")
