#!/usr/bin/env python
# manual

"""
This script allows you to manually control the simulator or Duckiebot
using the keyboard arrows.
"""
import sys
import argparse
import pyglet
from pyglet.window import key
import numpy as np
from duckietown_rl.gym_duckietown.simulator import Simulator
from duckietown_il._loggers import Logger
import matplotlib.pyplot as plt

import gym
# import gym_duckietown
# from gym_duckietown.envs import DuckietownEnv
# from gym_duckietown.wrappers import UndistortWrapper

# from experiments.utils import save_img


parser = argparse.ArgumentParser()
# parser.add_argument('--env-name', default=None)
parser.add_argument('--map-name', default='small_loop')
parser.add_argument('--distortion', default=False, action='store_true')
parser.add_argument('--draw-curve', action='store_true', help='draw the lane following curve')
parser.add_argument('--draw-bbox', action='store_true', help='draw collision detection bounding boxes')
parser.add_argument('--domain-rand', action='store_true', help='enable domain randomization')
parser.add_argument('--frame-skip', default=1, type=int, help='number of frames to skip')
parser.add_argument('--seed', default=1, type=int, help='seed')
args = parser.parse_args()


# if args.env_name and args.env_name.find('Duckietown') != -1:
#     env = DuckietownEnv(
#         seed = args.seed,
#         map_name = "udem1",
#         draw_curve = args.draw_curve,
#         draw_bbox = args.draw_bbox,
#         domain_rand = args.domain_rand,
#         frame_skip = args.frame_skip,
#         distortion = args.distortion,
#     )
# else:
#     env = gym.make(args.env_name)

env = Simulator(
        seed=123,  # random seed
        map_name="zigzag_dists",
        max_steps=500001,  # we don't want the gym to reset itself
        domain_rand=False,
        camera_width=640,
        camera_height=480,
        full_transparency=True
    )

logger = Logger(env, log_file=f'data.log')


env.reset()
env.render()

@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    """
    This handler processes keyboard commands that
    control the simulation
    """

    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print('RESET')
        env.reset()
        env.render()
    elif symbol == key.PAGEUP:
        env.unwrapped.cam_angle[0] = 0
    elif symbol == key.ESCAPE:
        env.close()
        sys.exit(0)

    # Take a screenshot
    # UNCOMMENT IF NEEDED - Skimage dependency
    # elif symbol == key.RETURN:
    #     print('saving screenshot')
    #     img = env.render('rgb_array')
    #     save_img('screenshot.png', img)

# Register a keyboard handler
key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)

def update(dt):
    """
    This function is called at every frame to handle
    movement/stepping and redrawing
    """

    action = np.array([0.0, 0.0])

    if key_handler[key.UP]:
        action = np.array([0.44, 0.44])
    if key_handler[key.DOWN]:
        action = np.array([-0.44, -0.44])
    if key_handler[key.LEFT]:
        action = np.array([0.2, +1])
    if key_handler[key.RIGHT]:
        action = np.array([0.2, -1])
    if key_handler[key.SPACE]:
        action = np.array([0, 0])

    # Speed boost
    if key_handler[key.LSHIFT]:
        action *= 1.5

    obs, reward, done, info = env.step(action)

    print('step_count = %s, reward=%.3f' % (env.unwrapped.step_count, reward))

    if key_handler[key.RETURN]:
        # from PIL import Image
        # im = Image.fromarray(obs)
        #
        # im.save('screen.png')
        exit(1)

    if done:
        print('done!')
        # logger.close()
        # env.reset()

    env.render()

    angle_d = info['Simulator']['lane_position']['angle_deg']
    dist = info['Simulator']['lane_position']['dist']

    fig, ax = plt.subplots(1, 1, constrained_layout=True)
    ax.imshow(obs)
    ax.set_title(f"Angle: [{angle_d:.3f}]\n"
                 f"Displacement: [{dist:.3f}]")
    # plt.show()
    plt.savefig(f"../duckietown_il/logs_360/log-{env.step_count}.png")


pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Enter main event loop
pyglet.app.run()

env.close()