from gym_duckietown.simulator import Simulator
from gym_duckietown.simulator import NotInLane
from utils.helpers import SteeringToWheelVelWrapper
import cv2
import numpy as np

# To convert to wheel velocities
wrapper = SteeringToWheelVelWrapper()

env = Simulator(
    seed=123,  # random seed
    map_name="zigzag_dists",  # "udem1",  #"zigzag_dists", "4way", "loop_empty","small_loop", "straight_road", "small_loop_cw",
    max_steps=5000001,
    domain_rand=True,
    camera_width=640,
    camera_height=480,
    accept_start_angle_deg=4,  # start close to straight
    full_transparency=True,
    distortion=True,
    randomize_maps_on_reset=False,
)

obs = env.reset()
#env.render()

EPISODES = 4
STEPS = 1000

speed_list = [0] * EPISODES
angle_list = [0] * EPISODES

k_p_values = list(range(9, 40))
k_d_values = list(range(8, 17))

data = np.zeros((len(k_p_values), len(k_d_values)))

CAMERA_FORWARD_DIST = 0.066
ROBOT_LENGTH = 0.18

for idx_p,p in enumerate(k_p_values):
    for idx_d,d in enumerate(k_d_values):
        rewards = 0
        for episode in range(0, EPISODES):
            for steps in range(0, STEPS):
                try:
                    lane_pose = env.get_lane_pos2(env.cur_pos, env.cur_angle)
                except NotInLane:
                    break

                distance_to_road_center = lane_pose.dist
                angle_from_straight_in_rads = lane_pose.angle_rad

                k_p = p
                k_d = d

                if -0.5 < lane_pose.angle_deg < 0.5:
                    speed = 1
                elif -1 < lane_pose.angle_deg < 1:
                    speed = 0.6
                elif -2 < lane_pose.angle_deg < 2:
                    speed = 0.5
                elif -10 < lane_pose.angle_deg < 10:
                    speed = 0.3
                else:
                    speed = 0.1

                # angle of the steering wheel, which corresponds to the angular velocity in rad/s
                steering = k_p*distance_to_road_center + k_d*angle_from_straight_in_rads
                action = wrapper.convert([speed, steering])
                obs, reward, done, info = env.step(action)

                dir_vec = env.get_dir_vec(env.cur_angle)
                center = env.cur_pos + (CAMERA_FORWARD_DIST - (ROBOT_LENGTH / 2)) * dir_vec
                try:
                    point, _ = env.closest_curve_point(env.cur_pos, env.cur_angle)
                    reward = np.dot((center - point).T, center - point)
                except TypeError:
                    reward = 0
                    pass

                #env.render()

                rewards += np.abs(reward)
            env.reset()
        data[idx_p][idx_d] = rewards


import pandas as pd
df = pd.DataFrame(data, columns=k_d_values, index=k_p_values)
df.to_csv("data2.csv")
