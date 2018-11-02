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
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import RMSprop, Adam, SGD
from keras.models import clone_model
from keras.callbacks import EarlyStopping
import gym

env = gym.make('NChain-v0')
print(env.reset())
print(env.step(1))
print(env.step(0))
print(env.step(0))
print(env.step(0))
print(env.step(0))
print(env.step(0))
print(env.step(0))
print(env.step(0))
print(env.step(0))
print(env.step(0))
print(env.step(0))
print(env.step(0))
print(env.step(0))
print(env.step(0))
print(env.step(0))
print(env.step(0))
print(env.step(0))
print(env.step(0))
print(env.step(0))
print(env.step(0))
print(env.step(0))
print(env.step(0))
print(env.step(0))
print(env.step(0))
print(env.step(0))
print(env.step(0))
print(env.step(0))
print(env.step(0))
print(env.step(0))
print(env.step(0))
print(env.step(0))
print(env.step(0))
print(env.step(0))
print(env.step(0))
print(env.step(0))
print(env.step(0))
print(env.step(0))