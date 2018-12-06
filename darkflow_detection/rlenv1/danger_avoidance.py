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
from keras.models import clone_model
from keras.callbacks import EarlyStopping
import gym
import rlenv1

# Define model
num_states = 5
num_actions = 4
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(num_states,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(num_actions, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['mae'])
model.summary()

# Train the model
env = gym.make('rlenv1-v0')
num_episodes = 30 #200
y = 0.95
eps = 0.6
decay_factor = 0.6
r_avg_list, num_crashes_list = [], []
for i in range(num_episodes):
    s = env.reset()
    eps *= decay_factor
    print("Episode {} of {}".format(i + 1, num_episodes))
    done, r_sum, count, num_crashes = False, 0, 0, 0
    while not done:
        if np.random.random() < eps:
            a = np.random.randint(0, num_actions)
        else:
            a = np.argmax(model.predict(np.identity(num_states)[s:s + 1]))
        new_s, r, done, _ = env.step(a, count)
        if new_s == num_states-1:
            num_crashes += 1
        target = r + y * np.max(model.predict(np.identity(num_states)[s:s + 1])) #new_s
        target_vec = model.predict(np.identity(num_states)[s:s + 1])[0]
        target_vec[a] = target
        model.fit(np.identity(num_states)[s:s + 1], target_vec.reshape(-1, num_actions), epochs=2, verbose=0)
        s = new_s
        r_sum += r
        count += 1
    print("reward:", r_sum)
    print("crashes:", num_crashes)
    r_avg_list.append(r_sum)
    num_crashes_list.append(num_crashes)

# Save model to file
model_json = model.to_json()
with open("rlmodel.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("rlmodel.h5")

# Plot results
fig, ax1 = plt.subplots()
t = range(1, num_episodes+1)
s1 = r_avg_list
ax1.plot(t, s1,'b')
ax1.set_xlabel('training episodes')
ax1.set_ylabel('cumulative attained reward', color='b')
ax1.tick_params('y', colors='b')
ax2 = ax1.twinx()
s2 = num_crashes_list
ax2.plot(t, s2, 'r')
ax2.set_ylabel('number of crashes', color='r')
ax2.tick_params('y', colors='r')
fig.tight_layout()
plt.show()
