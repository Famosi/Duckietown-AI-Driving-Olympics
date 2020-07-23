import sys
sys.path.append("../../")
from duckietown_rl.gym_duckietown.simulator import Simulator
from keras.models import load_model
from duckietown_rl.expert import Expert
import cv2
import os


# for macOS
os.environ['KMP_DUPLICATE_LIB_OK']='True'

env = Simulator(seed=123, map_name="zigzag_dists", max_steps=5000001, domain_rand=True, camera_width=640,
                camera_height=480, accept_start_angle_deg=1, full_transparency=True, distortion=True,
                randomize_maps_on_reset=True, draw_curve=False, draw_bbox=False, frame_skip=1, frame_rate=30)

model = load_model("trained_models/03_NVIDIA_actions.h5")

observation = env.reset()
cumulative_reward = 0.0
EPISODES = 3
STEPS = 200

expert = Expert(env=env)

observations = []
predictions = []
gts = []

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

        action = model.predict(observation.reshape(1, 60, 120, 3))[0]  # Predict
        # action_GT = expert.predict_action(env)  # Ground Truth

        observation, reward, done, info = env.step(action)

        observations.append(observation)
        predictions.append(action)
        # gts.append(action_GT)

        cumulative_reward += reward
        if done:
            env.reset()

        print(f"Reward: {reward:.2f}",
              f"\t| Action: [{action[0]:.3f}, {action[1]:.3f}]",
              f"\t| Speed: {env.speed:.2f}")

        cv2.imshow("obs", observation)
        cv2.waitKey(1)

    env.reset()

print('total reward: {}, mean reward: {}'.format(cumulative_reward, cumulative_reward // EPISODES))

env.close()

# Plot Predictions VS. Ground Truth
# fig = plt.figure(figsize=(40, 30))
# i = 0
# for prediction, gt, img in zip(predictions_25Ep, gts, observations):
#     fig, ax = plt.subplots(1, 1, constrained_layout=True)
#     ax.imshow(img)
#     ax.set_title(f"Pred: [{prediction[0]:.3f}, {prediction[1]:.3f}]\n"
#                  f"GT: [{gt[0]:.3f}, {gt[1]:.3f}]")
#     i += 1
#
# plt.show()
