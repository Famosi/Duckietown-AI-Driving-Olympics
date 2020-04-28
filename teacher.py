import numpy as np
import networkx as nx
from gym_duckietown.simulator import NotInLane
import copy

MAX = -100000
COF_LANE = 1000
COF_ALIGN = 10000
COF_SPEED = 100


class Expert:
    def __init__(self, env):
        self.env = env
        self.cof_lane = COF_LANE
        self.cof_align = COF_ALIGN
        self.cof_speed = COF_SPEED
        self.action_space = np.array([
            [1., 1.],
            [0.9, 1.],
            [1., 0.9],
            [0., 1.],
            [1., 0.]
        ])

    def predict_rollout(self, n, env):

        m = 0
        for i in range(0, n):
            m += self.action_space.shape[0] ** i

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
            copy.deepcopy(env.step_count),
            copy.deepcopy(env.timestamp)
        )
        next_parents = []
        rollout = nx.DiGraph()

        rollout.add_node(nodes[0], position=env.cur_pos, angle=env.cur_angle, speed=env.robot_speed, wheelVels=env.wheelVels, action_sequence=[],
                         node_sequence=[nodes[0]])
        nodes.pop(0)

        def __helper__(nodes, current_parent, next_parents, rollout, denv, robot_speed, cur_pos, cur_angle, state, last_action, wheelVels, delta_time, step_count, timestamp):
            if nodes:
                for action in range (self.action_space.shape[0]):
                    denv.set_env_params(robot_speed, cur_pos, cur_angle, state, last_action, wheelVels, delta_time, step_count, timestamp)
                    denv.step_rollout(self.action_space[action])

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
                            copy.deepcopy(denv.step_count),
                            copy.deepcopy(denv.timestamp)
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
                    return __helper__(nodes, current_parent, next_parents, rollout, current_parent[1],
                                      current_parent[2], current_parent[3], current_parent[4], current_parent[5],
                                      current_parent[6], current_parent[7], current_parent[8], current_parent[9],
                                      current_parent[10]
                                      )
                else:
                    return rollout
            else:
                return rollout

        if current_parent:
            return __helper__(nodes, current_parent, next_parents, rollout, current_parent[1], current_parent[2],
                              current_parent[3], current_parent[4], current_parent[5], current_parent[6],
                              current_parent[7], current_parent[8], current_parent[9], current_parent[10]
                              )

        return rollout

    def collect_rollout(self, dream_env):
        robot_speed = copy.deepcopy(dream_env.robot_speed)
        cur_pos = copy.deepcopy(dream_env.cur_pos)
        cur_angle = copy.deepcopy(dream_env.cur_angle)
        state = copy.deepcopy(dream_env.state)
        last_action = copy.deepcopy(dream_env.last_action)
        wheelVels = copy.deepcopy(dream_env.wheelVels)
        delta_time = copy.deepcopy(dream_env.delta_time)
        step_count = copy.deepcopy(dream_env.step_count)
        timestamp = copy.deepcopy(dream_env.timestamp)

        # predict 3 steps ahead
        rollout = self.predict_rollout(3, dream_env)

        # Reset the env to original one
        dream_env.set_env_params(robot_speed, cur_pos, cur_angle, state, last_action, wheelVels, delta_time, step_count,
                            timestamp)

        max_reward = MAX
        best_node = None
        for node in rollout.nodes:
            # if it's not the root
            if node > 1:
                # node's state of the agent
                position = rollout.nodes[node]['position']
                angle = rollout.nodes[node]['angle']

                try:
                    lane = self.env.get_lane_pos2(position, angle)
                except NotInLane:
                    break

                # Check if current action leads to a non-derivable tile
                if not dream_env.valid_pose_rollout(position, angle):
                    not_derivable = MAX
                else:
                    not_derivable = 0

                # Distance from the center of the lane, the smaller the better
                # dist = abs(lane.dist) * self.cof_lane
                dist = dream_env.dist_centerline_curve(position, angle)  # * self.cof_lane

                # Alignment with the lane, the higher the better
                align = lane.dot_dir  # * self.cof_align

                # Speed of the agent, the higher the better
                cur_action = rollout.nodes[node]['action_sequence'][-1]
                speed = sum(self.action_space[cur_action])  # * self.cof_speed

                # Calculate LOSS
                reward = (
                        + speed * self.cof_speed
                        + align * 100000
                        - dist * 10000
                        + not_derivable
                )

                # The node with the highest reward is the best node
                if reward > max_reward:
                    max_reward = reward
                    best_node = node

        # if there is no node from rollout take a random action
        if best_node is not None:
            action_seq = rollout.nodes[best_node]['action_sequence']
            action = self.action_space[action_seq[0]]
        else:
            action = np.random.randint(0, self.action_space.shape[0])
            action = self.action_space[action]

        return action

    def predict_action(self, dream_env):
        # Check if the agent is in a curve
        try:
            curve = dream_env._get_tile(
                dream_env.get_tile()[1][0],
                dream_env.get_tile()[1][1]
            )['kind'].startswith('curve')
        except ValueError:
            curve = False

        dist = dream_env.get_lane_pos2(dream_env.cur_pos, dream_env.cur_angle).dist

        if not curve:
            if abs(dist) < 0.05:
                self.cof_speed = COF_SPEED * 5
            if dist < -0.04:
                return [1., 0.9]
            elif dist > 0.04:
                return [0.9, 1.]
        else:
            self.cof_speed = COF_SPEED

        return self.collect_rollout(dream_env)
