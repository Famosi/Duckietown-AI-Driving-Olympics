import sys
sys.path.append("../../")
from duckietown_rl.gym_duckietown.simulator import Simulator
from keras.models import load_model
from duckietown_rl.expert import Expert
import matplotlib.pyplot as plt
import cv2
import os


# for macOS
os.environ['KMP_DUPLICATE_LIB_OK']='True'

env = Simulator(seed=123, map_name="zigzag_dists", max_steps=5000001, domain_rand=True, camera_width=640,
                camera_height=480, accept_start_angle_deg=4, full_transparency=True, distortion=True,
                randomize_maps_on_reset=True, draw_curve=False, draw_bbox=False, frame_skip=2, frame_rate=15)

model = load_model("trained_models/01_NVIDIA.h5")

observation = env.reset()
env.render()
cumulative_reward = 0.0
EPISODES = 1
STEPS = 80

observations = []
actions_predict = []

expert = Expert(env=env)
actions_gt = []

for episode in range(0, EPISODES):
    for steps in range(0, STEPS):
        # Cut the horizon: obs.shape = (480,640,3) --> (300,640,3)
        observation = observation[150:450, :]
        # we can resize the image here
        observation = cv2.resize(observation, (120, 60))
        # NOTICE: OpenCV changes the order of the channels !!!
        observation = cv2.cvtColor(observation, cv2.COLOR_BGR2RGB)
        # Rescale the image
        observation = observation * 1.0/255

        action = model.predict(observation.reshape(1, 60, 120, 3))[0]
        # action = expert.predict_action(env)
        observation, reward, done, info = env.step(action)

        observations.append(observation)
        actions_predict.append(action)
        actions_gt.append(expert.predict_action(env))

        cumulative_reward += reward
        if done:
            env.reset()

        print(f"Reward: {reward:.2f}",
              f"\t| Action: [{action[0]:.3f}, {action[1]:.3f}]",
              f"\t| Speed: {env.speed:.2f}")

        # cv2.imshow("obs", observation)
        # if cv2.waitKey() & 0xFF == ord('q'):
        #     break

        env.render()
    env.reset()

print('total reward: {}, mean reward: {}'.format(cumulative_reward, cumulative_reward // EPISODES))

env.close()
# model.close()

fig = plt.figure(figsize=(40, 30))
i = 0
for action_predict, action_gt, img in zip(actions_predict, actions_gt, observations):
    ax = fig.add_subplot(12, 10, i+1, xticks=[], yticks=[])
    ax.imshow(img)
    ax.set_title(f"Action_predict: [{action_predict[0]:.1f}, {action_predict[1]:.1f}]\n"
                 f"GT: [{action_gt[0]:.1f}, {action_gt[1]:.1f}]")
    i += 1

plt.show()
