import cv2
import time
from duckietown_rl.expert import Expert
from duckietown_il.env import launch_env
from duckietown_il._loggers import Logger

env = launch_env()

EPISODES, STEPS = 50, 250

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

        logger.log(observation, action, reward, done, info)

    logger.on_episode_done()  # speed up logging by flushing the file
    env.reset()

logger.close()
env.close()

end_time = time.time()

print(f"Process finished. It took {(end_time - start_time) / (60*60):.2f} hours!")

