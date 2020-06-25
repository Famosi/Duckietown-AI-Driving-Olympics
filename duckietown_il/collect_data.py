import cv2
import time
from expert import Expert
from env import launch_env
from _loggers import Logger
import numpy as np


angle_intervals = [-0.24, -0.18, -0.12, -0.06, -0.01, 0.01, 0.06, 0.12, 0.18, 0.24]
disp_intervals = [-0.10, -0.08, -0.05, -0.03, -0.01, 0.01, 0.03, 0.05, 0.08, 0.10]


# Return TRUE if don't have enough cur_angle OR cur_displacement
# TRUE otherwise
def check_intervals(angles, displacements, cur_angle, cur_disp):
    check_angle = True
    check_disp = True
    for x, y in zip(angle_intervals, angle_intervals[1:]):
        if x < cur_angle < y and np.histogram(angles, bins=(x, y))[0] > 3000:
            check_angle = False
    for x, y in zip(disp_intervals, disp_intervals[1:]):
        if x < cur_disp < y and np.histogram(displacements, bins=(x, y))[0] > 3000:
            check_disp = False
    return check_angle or check_disp


# Return TRUE if we have, for all intervals, more than 2500 samples
def is_log(angles, displacements):
    angle_is_log = True
    disp_is_log = True
    for x, y in zip(angle_intervals, angle_intervals[1:]):
        if np.histogram(angles, bins=(x, y))[0] < 2500:
            angle_is_log = False
    for x, y in zip(disp_intervals, disp_intervals[1:]):
        if np.histogram(displacements, bins=(x, y))[0] < 2500:
            disp_is_log = False
    return angle_is_log and disp_is_log


env = launch_env()

EPISODES, STEPS = 1, 4

logger = Logger(env, log_file=f'train-{int(EPISODES*STEPS/1000)}k.log')

expert = Expert(env=env)

angles = []
displacements = []
pts = []
observations = []

start_time = time.time()
print(f"[INFO]Starting to get logs for {EPISODES} episodes each {STEPS} steps..")
# let's collect our samples
for episode in range(0, EPISODES):
    for step in range(0, STEPS):
        # we use our 'expert' to predict the next action.
        action = expert.predict_action(env)

        observation, reward, done, info = env.step(action)

        # Cut the horizon: obs.shape = (480,640,3) --> (300,640,3)
        observation = observation[150:450, :]
        # we can resize the image here
        observation = cv2.resize(observation, (120, 60))
        # NOTICE: OpenCV changes the order of the channels !!!
        observation = cv2.cvtColor(observation, cv2.COLOR_BGR2RGB)

        if done:
            print(f"#Episode: {episode}\t | #Step: {step}")
            break

        lp = env.get_lane_pos2(env.cur_pos, env.cur_angle)
        angles.append(lp.angles)
        displacements.append(lp.dist)

        pts.append(env.get_pts())
        observations.append(observation)

        # logger.log(observation, action, reward, done, info)


        # if check_intervals(angles, displacements, lp.angles, lp.dist):
        #     logger.log(observation, action, reward, done, info)
        # if not is_log(angles, displacements) and EPISODES-episode == 0:
        #     episode -= 1

    logger.on_episode_done()  # speed up logging by flushing the file
    env.reset()

logger.close()
env.close()

end_time = time.time()
print(pts)
print(f"Process finished. It took {(end_time - start_time) / (60*60):.2f} hours!")
