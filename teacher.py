import numpy as np
import networkx as nx
from gym_duckietown.simulator import NotInLane
import matplotlib.pyplot as plt
import copy

MAX = -100000
# parameters for the pure pursuit controller
POSITION_THRESHOLD = 0.04
REF_VELOCITY = 0.1
GAIN = 10
FOLLOWING_DISTANCE = 0.3


class PurePursuitExpert:
    def __init__(self, env, actions, ref_velocity=REF_VELOCITY, position_threshold=POSITION_THRESHOLD,
                 following_distance=FOLLOWING_DISTANCE, max_iterations=1000):
        self.env = env
        self.actions = actions
        self.following_distance = following_distance
        self.max_iterations = max_iterations
        self.ref_velocity = ref_velocity
        self.position_threshold = position_threshold

    def predict_rollout_head(self, n, env):

        m = 0
        for i in range(0, n):
            m += self.actions.shape[0] ** i

        nodes = list(range(1, m + 1))

        current_parent = (
            nodes[0],
            env,
            copy.deepcopy(env.robot_speed),
            copy.deepcopy(env.cur_pos),
            copy.deepcopy(env.cur_angle),
            copy.deepcopy(env.state),
            copy.deepcopy(env.last_action),
            copy.deepcopy(env.wheelVels),
            copy.deepcopy(env.delta_time),
            copy.deepcopy(env.step_count)
        )
        next_parents = []
        rollout = nx.DiGraph()

        rollout.add_node(nodes[0], position=env.cur_pos, angle=env.cur_angle, speed=env.robot_speed, wheelVels=env.wheelVels, action_sequence=[],
                         node_sequence=[nodes[0]])
        nodes.pop(0)

        def __helper__(nodes, current_parent, next_parents, rollout, denv, robot_speed, cur_pos, cur_angle, state, last_action, wheelVels, delta_time, step_count):
            if nodes:
                for action in range (self.actions.shape[0]):
                    denv.set_env_params(robot_speed, cur_pos, cur_angle, state, last_action, wheelVels, delta_time, step_count)
                    denv.step_rollout(self.actions[action])

                    dream_position = denv.cur_pos
                    dream_angle = denv.cur_angle
                    dream_speed = denv.robot_speed
                    dream_wheelVels = denv.wheelVels

                    next_parents.append(
                        (
                            nodes[0],
                            denv,
                            copy.deepcopy(denv.robot_speed),
                            copy.deepcopy(denv.cur_pos),
                            copy.deepcopy(denv.cur_angle),
                            copy.deepcopy(denv.state),
                            copy.deepcopy(denv.last_action),
                            copy.deepcopy(denv.wheelVels),
                            copy.deepcopy(denv.delta_time),
                            copy.deepcopy(env.step_count)
                        )
                    )

                    current_action_sequence = rollout.nodes[current_parent[0]]["action_sequence"].copy()
                    current_action_sequence.append(action)
                    current_node_sequence = rollout.nodes[current_parent[0]]["node_sequence"].copy()
                    current_node_sequence.append(nodes[0])
                    rollout.add_node(nodes[0], position=np.array(dream_position), angle=dream_angle, speed=dream_speed, wheelVels=dream_wheelVels,
                                     action_sequence=current_action_sequence,
                                     node_sequence=current_node_sequence)
                    rollout.add_edge(current_parent[0], nodes[0], action=action)
                    nodes.pop(0)

                current_parent = next_parents.pop(0)

                if current_parent:
                    return __helper__(nodes, current_parent, next_parents, rollout, current_parent[1], current_parent[2], current_parent[3], current_parent[4], current_parent[5], current_parent[6], current_parent[7], current_parent[8], current_parent[9])
                else:
                    return rollout
            else:
                return rollout

        if current_parent:
            return __helper__(nodes, current_parent, next_parents, rollout, current_parent[1], current_parent[2], current_parent[3], current_parent[4], current_parent[5], current_parent[6], current_parent[7], current_parent[8], current_parent[9])

        return rollout

    def dream_forward(self, dream_env):
        robot_speed = copy.deepcopy(dream_env.robot_speed)
        cur_pos = copy.deepcopy(dream_env.cur_pos)
        cur_angle = copy.deepcopy(dream_env.cur_angle)
        state = copy.deepcopy(dream_env.state),
        last_action = copy.deepcopy(dream_env.last_action),
        wheelVels = copy.deepcopy(dream_env.wheelVels),
        delta_time = copy.deepcopy(dream_env.delta_time)
        step_count = copy.deepcopy(dream_env.step_count)

        # predict 3 steps ahead
        rollout = self.predict_rollout_head(3, dream_env)

        dream_env.set_env_params(robot_speed, cur_pos, cur_angle, state[0], last_action, wheelVels, delta_time, step_count)

        curve = dream_env._get_tile(dream_env.get_tile()[1][0], dream_env.get_tile()[1][1])['kind'].startswith('curve')
        dist = dream_env.get_lane_pos2(dream_env.cur_pos, dream_env.cur_angle).dist

        min_loss = MAX
        best_node = None
        for node in rollout.nodes:
            # if it's it's not the root
            if node > 1:
                position = rollout.nodes[node]['position']
                angle = rollout.nodes[node]['angle']

                try:
                    lane = self.env.get_lane_pos2(position, angle)
                except NotInLane:
                    break

                # LOSS-1: distance to the center of the lane
                # dist_lane = dream_env.dist_centerline_curve(position, angle)
                dist_lane = abs(lane.dist)

                # LOSS-2: the direction of the agent in relation to the (direction of the) lane
                angle_deg = np.abs(lane.angle_deg)

                # Calculate LOSS
                action = rollout.nodes[node]['action_sequence'][0]
                speed = sum(self.actions[action])

                if not dream_env.valid_pose_rollout(position, angle):
                    not_derivable = MAX
                else:
                    not_derivable = 0

                if not curve:
                    coef_speed = 50
                else:
                    coef_speed = 10

                loss = (
                        + coef_speed*speed
                        # - 10*angle_deg
                        + 10000*lane.dot_dir
                        - 1000*dist_lane
                        + not_derivable
                )

                if loss > min_loss:
                    min_loss = loss
                    best_node = node

        if best_node is not None:
            action_seq = rollout.nodes[best_node]['action_sequence']
            next_action = self.actions[action_seq[0]]
        else:
            print("RANDOM")
            action = np.random.randint(0, self.actions.shape[0])
            next_action = self.actions[action]

        if not curve and dist < -0.07:
            next_action = [1., 0.95]
        if not curve and dist > 0.07:
            next_action = [0.9, 1.]

        return next_action

    def align(self, env):
        closest_point, closest_tangent = self.env.closest_curve_point(self.env.cur_pos, self.env.cur_angle)
        if closest_point is None:
            return 0.0, 0.0  # Should return done in the environment

        iterations = 0
        lookup_distance = self.following_distance
        curve_point = None
        while iterations < self.max_iterations:
            # Project a point ahead along the curve tangent,
            # then find the closest point to to that
            follow_point = closest_point + closest_tangent * lookup_distance
            curve_point, _ = self.env.closest_curve_point(follow_point, self.env.cur_angle)

            # If we have a valid point on the curve, stop
            if curve_point is not None:
                break

            iterations += 1
            lookup_distance *= 0.5

        # Compute a normalized vector to the curve point
        point_vec = curve_point - self.env.cur_pos
        point_vec /= np.linalg.norm(point_vec)

        dot = np.dot(self.env.get_right_vec(), point_vec)
        steering = GAIN * -dot

        return self.ref_velocity, steering

