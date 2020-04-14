import cv2
from env import launch_env
from teacher import Expert
from _loggers import Logger
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import pickle
import time
from sklearn.preprocessing import MinMaxScaler

env = launch_env()
# logger = Logger(env, log_file='train.log')

reward_acc = np.array([])

left_velocity = np.array([])
right_velocity = np.array([])

DEBUG = True

rewards = 0
STEPS = 200
EPISODES = 10
expert = Expert(env=env)

# let's collect our samples
for episode in range(0, EPISODES):
    print("Episode:", episode+1)
    for step in range(0, STEPS):
        # we use our 'expert' to predict the next action.
        action = expert.predict_action(env)

        observation, reward, done, info = env.step(action)

        if done:
            break

        left_velocity = np.append(left_velocity, action[0])
        right_velocity = np.append(right_velocity, action[1])

        reward_acc = np.append(reward_acc, reward)
        rewards += reward

        # we can resize the image here
        observation = cv2.resize(observation, (80, 60))
        # NOTICE: OpenCV changes the order of the channels !!!
        observation = cv2.cvtColor(observation, cv2.COLOR_BGR2RGB)

        # we may use this to debug our expert.
        if DEBUG:
            cv2.imshow('debug', observation)
            cv2.waitKey(1)
        # cv2.imshow("obs", observation)
        # if cv2.waitKey() & 0xFF == ord('q'):
        #     break

        # logger.log(observation, action, reward, done, info)
        # [optional] env.render() to watch the expert interaction with the environment
        # we log here
    # logger.on_episode_done()  # speed up logging by flushing the file
    env.reset()

# logger.close()
env.close()

print("TOTAL REWARD:", rewards)

plt.subplot(2, 1, 1)
plt.plot(reward_acc, color='red', linewidth=2)
plt.title("reward")

plt.subplot(2, 1, 2)
plt.plot(left_velocity, color="green", linewidth=2)
plt.plot(right_velocity, color="orange", linewidth=2)
plt.title("L(g)/R(o) Vel")

plt.show()
