from duckietown_il._loggers import Reader
import matplotlib.pyplot as plt
import numpy as np

reader = Reader('train-100k.log')

observations, actions, cur_angles, info = reader.read()

observations = np.array(observations)
actions = np.array(actions)
cur_angles = np.array(cur_angles)
angle_rad = np.array([i['Simulator']['lane_position']['angle_rad'] for i in info])
angle_deg = np.array([i['Simulator']['lane_position']['angle_deg'] for i in info])
displacement = np.array([i['Simulator']['lane_position']['dist'] for i in info])
speed = np.array([i['Simulator']['robot_speed'] for i in info])
info = np.array(info)

num_bins = 11
plt.figure(figsize=(30,24))

plt.subplot(321)
colors = ["blue", "orange"]
labels = ["left", "right"]
n, bins, patches = plt.hist(actions, num_bins, color=colors, alpha=0.5, label=labels, stacked=False)
plt.title("Histogram of $Wheel Velocities$")
plt.xlabel("Wheel Velocity")
plt.ylabel("Counts")
plt.legend()

plt.subplot(322)
n, bins, patches = plt.hist(speed, num_bins, alpha=0.5, color="blue")
plt.title("Histogram of the $Speed$ of the car")
plt.xlabel("Speed in m/s")
plt.ylabel("Counts")

plt.subplot(323)
n, bins, patches = plt.hist(angle_deg, num_bins, alpha=0.5, color="red")
plt.xlabel("Angles in degrees")
plt.ylabel("Counts")

plt.subplot(324)
n, bins, patches = plt.hist(angle_rad, num_bins, alpha=0.5, color="red")
plt.xlabel("Angles in radians")
plt.ylabel("Counts")

plt.subplot(325)
n, bins, patches = plt.hist(displacement, num_bins, alpha=0.5, color="green")
plt.title("Histogram of Distance to the center line")
plt.xlabel("Distance")
plt.ylabel("Counts")
plt.show()