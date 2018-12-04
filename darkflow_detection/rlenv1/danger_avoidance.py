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

# print(env.reset())
# print(env.step(1))
# print(env.step(0))

# Reward matrix:
# State 1: dot outside
# 	Stay: +100
# 	Move down: -50
# 	Move left: -100
# 	Move right: -100
# State 2: dot inside, not big enough
# 	Stay: -100
# 	Move down: +100
# 	Move left: +20
# 	Move right: +20
# State 3: dot inside, big (accident about to happen)
# 	Stay: -100
# 	Move down: +50
# 	Move left: +100
# 	Move right: +100


# def naive_sum_reward_agent(env, num_episodes=1000):
# 	# this is the table that will hold our summated rewards for
# 	# each action in each state
# 	r_table = np.zeros((5, 2))
# 	for g in range(num_episodes):
# 		s = env.reset()
# 		done = False
# 		while not done:
# 			if np.sum(r_table[s, :]) == 0:
# 				# make a random selection of actions
# 				a = np.random.randint(0, 2)
# 			else:
# 				# select the action with highest cummulative reward
# 				a = np.argmax(r_table[s, :])
# 			new_s, r, done, _ = env.step(a)
# 			r_table[s, a] += r
# 			s = new_s
# 	return r_table

# def q_learning_with_table(env, num_episodes=1000):
# 	q_table = np.zeros((5, 2))
# 	y = 0.95
# 	lr = 0.8
# 	for i in range(num_episodes):
# 		s = env.reset()
# 		done = False
# 		while not done:
# 			if np.sum(q_table[s,:]) == 0:
# 				# make a random selection of actions
# 				a = np.random.randint(0, 2)
# 			else:
# 				# select the action with largest q value in state s
# 				a = np.argmax(q_table[s, :])
# 			new_s, r, done, _ = env.step(a)
# 			q_table[s, a] += r + lr*(y*np.max(q_table[new_s, :]) - q_table[s, a])
# 			s = new_s
# 	return q_table

# def eps_greedy_q_learning_with_table(env, num_episodes=1000):
# 	q_table = np.zeros((5, 2))
# 	y = 0.95
# 	eps = 0.5
# 	lr = 0.8
# 	decay_factor = 0.999
# 	for i in range(num_episodes):
# 		s = env.reset()
# 		eps *= decay_factor
# 		done = False
# 		while not done:
# 			# select the action with highest cummulative reward
# 			if np.random.random() < eps or np.sum(q_table[s, :]) == 0:
# 				a = np.random.randint(0, 2)
# 			else:
# 				a = np.argmax(q_table[s, :])
# 			# pdb.set_trace()
# 			new_s, r, done, _ = env.step(a)
# 			q_table[s, a] += r + lr * (y * np.max(q_table[new_s, :]) - q_table[s, a])
# 			s = new_s
# 	return q_table

# # print(eps_greedy_q_learning_with_table(env))

# model = Sequential()
# # model.add(InputLayer(batch_input_shape=(1, 4)))
# model.add(Dense(16, activation='relu', input_shape=(4,)))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(4, activation='softmax'))
# model.compile(loss='mse', optimizer='adam', metrics=['mae'])
# model.summary()

#def make_rl():
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(4,)))
# model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dense(4, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['mae'])
model.summary()

env = gym.make('rlenv1-v0')
num_episodes = 200
y = 0.95
eps = 0.001
decay_factor = 0.999
r_avg_list = []
for i in range(num_episodes):
    s = env.reset()
    eps *= decay_factor
    if i % 1 == 0:
        print("Episode {} of {}".format(i + 1, num_episodes))
    done = False
    r_sum = 0
    count = 0
    while not done and count <= 10000:
        if np.random.random() < eps:
            a = np.random.randint(0, 4)
        else:
            a = np.argmax(model.predict(np.identity(4)[s:s + 1]))
        new_s, r, done, _ = env.step(a)
        # if done:
        #     break
        target = r + y * np.max(model.predict(np.identity(4)[s:s + 1])) #new_s
        target_vec = model.predict(np.identity(4)[s:s + 1])[0]
        # print(target_vec)
        target_vec[a] = target
        # print(target_vec)
        model.fit(np.identity(4)[s:s + 1], target_vec.reshape(-1, 4), epochs=1, verbose=0)
        s = new_s
        r_sum += r
        count += 1
    print(r_sum)
    print(count)
    r_avg_list.append(r_sum)

# print(r_avg_list)
plt.plot(r_avg_list)
plt.show()

return env