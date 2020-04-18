from gym_duckietown.simulator import Simulator


def launch_env():
    env = Simulator(
        seed=123,  # random seed
        map_name="4way",
        max_steps=500001,  # we don't want the gym to reset itself
        camera_width=640,
        camera_height=480,
        draw_curve=False,
        frame_skip=2,
        frame_rate=15,
        accept_start_angle_deg=4,  # @simone start close to straight
        full_transparency=True,
        distortion=False,
        randomize_maps_on_reset=True,
        draw_bbox=True
    )

    return env
