from gym_duckietown.simulator import Simulator


def launch_env():
    env = Simulator(
        seed=123,  # random seed
        map_name="udem1",
        max_steps=500001,  # we don't want the gym to reset itself
        domain_rand=True,
        camera_width=640,
        camera_height=480,
        draw_curve=False,
        frame_skip=1,
        frame_rate=10,
        accept_start_angle_deg=4,  # @simone start close to straight
        full_transparency=True,
        distortion=False,
        randomize_maps_on_reset=False
    )

    return env
