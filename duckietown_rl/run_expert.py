import cv2
from duckietown_rl.env import launch_env
from duckietown_rl.expert import Expert

env = launch_env()
DEBUG = True

STEPS = 200
EPISODES = 1

expert = Expert(env=env)

# let's collect our samples
for episode in range(0, EPISODES):
    for step in range(0, STEPS):
        # we use our 'expert' to predict the next action.
        action = expert.predict_action(env)

        observation, reward, done, info = env.step(action)

        if done:
            break

        # we may use this to debug our expert.
        if DEBUG:
            cv2.imshow('debug', observation)
            cv2.waitKey(1)

    env.reset()

env.close()

