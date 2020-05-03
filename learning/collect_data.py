import cv2
from .env import launch_env
from .expert import Expert
from ._loggers import Logger
import numpy as np
import matplotlib.pyplot as plt

env = launch_env()
# logger = Logger(env, log_file='train.log')

DEBUG = False

STEPS = 200
EPISODES = 5

expert = Expert(env=env)

# let's collect our samples
for episode in range(0, EPISODES):
    for step in range(0, STEPS):
        # we use our 'expert' to predict the next action.
        action = expert.predict_action(env)

        observation, reward, done, info = env.step(action)

        if done:
            break

        # we can resize the image here
        observation = cv2.resize(observation, (80, 60))
        # NOTICE: OpenCV changes the order of the channels !!!
        observation = cv2.cvtColor(observation, cv2.COLOR_BGR2RGB)

        # we may use this to debug our expert.
        if DEBUG:
            cv2.imshow('debug', observation)
            cv2.waitKey(1)

        # logger.log(observation, action, reward, done, info)
        # env.render()  # to watch the expert interaction with the environment
        # we log here
    # logger.on_episode_done()  # speed up logging by flushing the file
    env.reset()

# logger.close()
env.close()

