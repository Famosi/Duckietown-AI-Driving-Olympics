import cv2
import time
from expert import Expert
from env import launch_env
from _loggers import Logger
import numpy as np
import matplotlib.pyplot as plt

#
# angle_intervals = [-0.24, -0.18, -0.12, -0.06, -0.01, 0.01, 0.06, 0.12, 0.18, 0.24]
# disp_intervals = [-0.10, -0.08, -0.05, -0.03, -0.01, 0.01, 0.03, 0.05, 0.08, 0.10]


# Return TRUE if don't have enough cur_angle OR cur_displacement
# TRUE otherwise
# def check_intervals(angles, displacements, cur_angle, cur_disp):
#     check_angle = True
#     check_disp = True
#     for x, y in zip(angle_intervals, angle_intervals[1:]):
#         if x < cur_angle < y and np.histogram(angles, bins=(x, y))[0] > 3000:
#             check_angle = False
#     for x, y in zip(disp_intervals, disp_intervals[1:]):
#         if x < cur_disp < y and np.histogram(displacements, bins=(x, y))[0] > 3000:
#             check_disp = False
#     return check_angle or check_disp
#
#
# # Return TRUE if we have, for all intervals, more than 2500 samples
# def is_log(angles, displacements):
#     angle_is_log = True
#     disp_is_log = True
#     for x, y in zip(angle_intervals, angle_intervals[1:]):
#         if np.histogram(angles, bins=(x, y))[0] < 2500:
#             angle_is_log = False
#     for x, y in zip(disp_intervals, disp_intervals[1:]):
#         if np.histogram(displacements, bins=(x, y))[0] < 2500:
#             disp_is_log = False
#     return angle_is_log and disp_is_log


env = launch_env()

EPISODES, STEPS = 100, 100

logger = Logger(env, log_file=f'train-{int(EPISODES*STEPS/1000)}k.log')

# logger = Logger(env, log_file=f'data-{int(EPISODES*STEPS)}.log')

expert = Expert(env=env)

angles = []
displacements = []
pts = []
positions = []
positions_x = []
positions_y = []

observations = []

# env.render()
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

        # lp = env.get_lane_pos2(env.cur_pos, env.cur_angle)
        # angles.append(lp.angle_deg)
        # displacements.append(lp.dist)
        #
        pts.append(env.get_pts())
        positions.append(env.cur_pos)

        # observations.append(observation)

        # env.render()

        pts = env.get_pts()
        pts_prev = pts[:10]
        pts_cur = pts[10:20]
        pts_next = pts[20:30]

        logger.log(observation, action, pts_prev, pts_cur, pts_next, reward, done, info)

        # cv2.imshow("obs", observation)

        # if check_intervals(angles, displacements, lp.angles, lp.dist):
        #     logger.log(observation, action, reward, done, info)
        # if not is_log(angles, displacements) and EPISODES-episode == 0:
        #     episode -= 1

    logger.on_episode_done()  # speed up logging by flushing the file
    env.reset()

logger.close()
env.close()

end_time = time.time()

print(f"Process finished. It took {(end_time - start_time) / (60*60):.2f} hours!")

# pts_new = []
# for i in range(0, len(pts)):
#     for j in range(0, len(pts[i])):
#         pts_new.append(pts[i][j])
#
# pts_x = []
# pts_y = []
# for i in range(len(pts_new)):
#     pts_x.append((pts_new[i][0]))
#     pts_y.append((pts_new[i][2]))
#
# fig = plt.figure(figsize=(10, 10))
# i = 0
# for pts_i, pos in zip(pts, positions):
#     ax = fig.add_subplot(5, 5, i+1)
#     ax.scatter(pos[0], pos[2], c='blue')
#     for j in range(0, len(pts_i)):
#         ax.scatter(pts_i[j][0], pts_i[j][2], c='red')
#     i += 1
#
# plt.show()
