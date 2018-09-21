import os
import cv2
import time
import argparse
import multiprocessing
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import keras
from keras import backend as K
from keras.datasets import mnist, cifar10
from keras.models import Sequential, load_model, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import RMSprop, Adam, SGD
from keras.models import clone_model
from keras.callbacks import EarlyStopping
# from yolo_utils import *
# from darkflow.net.build import TFNet
import re

class darkflow_prediction():

	def __init__(self, image_file):
		self.image = cv2.imread(image_file, 1)
		print(type(self.image))

pred = darkflow_prediction("cars1.png")
