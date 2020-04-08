import cv2
from env import launch_env
from teacher import PurePursuitExpert
from _loggers import Logger
from utils.helpers import SteeringToWheelVelWrapper
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import pickle
import time
from sklearn.preprocessing import MinMaxScaler

env = launch_env()

# To convert to wheel velocities
wrapper = SteeringToWheelVelWrapper()
expert = PurePursuitExpert(env=env)

# logger = Logger(env, log_file='train-new-controller.log')
reward_acc = np.array([])

left_velocity = np.array([])
right_velocity = np.array([])

positions_x = np.array([])
positions_y = np.array([])

actions = np.array([])

dot_dirs = np.array([])

with open('./tracks/track_' + str(1439) + '.pkl', "rb") as pickle_file:
    df = pickle.load(pickle_file)
    line = [tuple(r) for r in df.to_numpy().tolist()]

xs = []
ys = []
for point in line:
    xs.append(point[:1][0])
    ys.append(point[1:3][0])

track = [xs, ys]

STEPS = 500
EPISODES = 5

DEBUG = True

rewards = 0
# let's collect our samples
for episode in range(0, EPISODES):
    for step in range(0, STEPS):
        # we use our 'expert' to predict the next action.
        # action = expert.predict(step, env)
        action = expert.dream_forward(env, track)

        left_velocity = np.append(left_velocity, action[0])
        right_velocity = np.append(right_velocity, action[1])

        observation, reward, done, info = env.step(action)

        actions = np.append(actions, action)

        positions_x = np.append(positions_x, env.cur_pos[0])
        positions_y = np.append(positions_y, env.cur_pos[2])

        reward_acc = np.append(reward_acc, reward)
        rewards += reward

        # try:
        #     lane = env.get_lane_pos2(env.cur_pos, env.cur_angle)
        #     dot_dirs = np.append(dot_dirs, np.abs(lane.dot_dir))
        # except Exception as e:
        #     print(e)
        #     break

        closest_point, _ = env.closest_curve_point(env.cur_pos, env.cur_angle)

        if closest_point is None:
            done = True
            break

        # we can resize the image here
        observation = cv2.resize(observation, (80, 60))
        # NOTICE: OpenCV changes the order of the channels !!!
        observation = cv2.cvtColor(observation, cv2.COLOR_BGR2RGB)

        # we may use this to debug our expert.
        if DEBUG:
            cv2.imshow('debug', observation)
            cv2.waitKey(1)
            # if cv2.waitKey() & 0xFF == ord('q'):
            #    break

        # logger.log(observation, action, reward, done, info)
        # [optional] env.render() to watch the expert interaction with the environment
        # we log here

    # logger.on_episode_done()  # speed up logging by flushing the file
    # env.reset()

# logger.close()
env.close()

print("TOTAL REWARD:", rewards)

# scaler = MinMaxScaler()
# dot_dirs = scaler.fit_transform(dot_dirs.reshape(-1, 1))
# plt.plot(left_velocity, label="left")
# plt.plot(right_velocity, label="right")
try:
    plt.subplot(2, 1, 1)
    plt.plot(reward_acc, color='red')
    plt.title("reward")

    plt.subplot(2, 1, 2)
    plt.plot(left_velocity, color="green")
    plt.plot(right_velocity, color="orange")
    plt.title("L(g)/R(o) Vel")

    # plt.subplot(2, 1, 2)
    # plt.scatter(track[0], track[1], color="black", s=2)
    # plt.scatter(positions_x, positions_y, color="green", s=5)
    # plt.title("Positions on TRACK")

    plt.show()
except Exception:
    print("reward", reward_acc)
    print("left", left_velocity)
    print("right", right_velocity)

""" ======================================================================================== 
Generate and save a track as a .pkl file
For debugging purpose 
======================================================================================== """

# def save_track(track):
#     ep = pd.DataFrame(track, columns=["position_x", "position_y", "shift_x", "shift_y"])
#     ep.to_pickle("./tracks/track_" + str(ep["position_x"].shape[0]) + ".pkl")

# track_x = []
# track_y = []
#
# for idx, _ in enumerate(env.pts):
#     for point in env.pts[idx]:
#             track_x.append(point[0])
#             track_y.append(point[2])
#
# track_half = list(zip(track_x, track_y))
# track_half = sorted(track_half, key=lambda k: k[0] + k[1])
#
# half = []
# for i, _ in enumerate(track_half):
#     if i % 2:
#         half.append(track_half[i])
#
# half = [list(t) for t in zip(*half)]
#
# track_x = half[0]
# track_y = half[1]
# shift_x = [(j - i) for i, j in zip(track_x, track_x[1:])]
# shift_y = [(j - i) for i, j in zip(track_y, track_y[1:])]
#
# track = list(zip(track_x, track_y, shift_x, shift_y))
# save_track(track)
#
# plt.scatter(track_x, track_y, color='red', s=3)
# plt.show()
#
# plt.plot(env.pts)
#
# plt.plot(pos_curve_dist, label="distance")
# plt.scatter(positions_x, positions_y, label="positions", s=3)
# plt.scatter(xs, ys, label="track", s=3)
# plt.show()
