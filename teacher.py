import numpy as np
import networkx as nx
from gym_duckietown.simulator import NotInLane
import matplotlib.pyplot as plt
import copy

MIN = 100000
MAX = -10000000


class PurePursuitExpert:
    def __init__(self, env):
        self.env = env.unwrapped
        self.actions = np.array([
            [1., 1.],
            [0.4, 1.],
            [1., 0.4],
            # [0.6, 1.],
            # [1., 0.6]
            [0.8, 1.],
            [1., 0.8]
        ])

    def predict_rollout_head(self, n, env):

        m = 0
        for i in range(0, n+1):
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
            copy.deepcopy(env.delta_time)
        )
        next_parents = []
        rollout = nx.DiGraph()

        rollout.add_node(nodes[0], position=env.cur_pos, angle=env.cur_angle, speed=env.robot_speed, wheelVels=env.wheelVels, action_sequence=[],
                         node_sequence=[nodes[0]])
        nodes.pop(0)

        def __helper__(nodes, current_parent, next_parents, rollout, denv, robot_speed, cur_pos, cur_angle, state, last_action, wheelVels, delta_time):
            if nodes:
                for action in range (self.actions.shape[0]):
                    denv.set_env_params(robot_speed, cur_pos, cur_angle, state, last_action, wheelVels, delta_time)
                    denv.step(self.actions[action])

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
                            copy.deepcopy(denv.delta_time)
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
                    return __helper__(nodes, current_parent, next_parents, rollout, current_parent[1], current_parent[2], current_parent[3], current_parent[4], current_parent[5], current_parent[6], current_parent[7], current_parent[8])
                else:
                    return rollout
            else:
                return rollout

        if current_parent:
            return __helper__(nodes, current_parent, next_parents, rollout, current_parent[1], current_parent[2], current_parent[3], current_parent[4], current_parent[5], current_parent[6], current_parent[7], current_parent[8])

        return rollout

    def dream_forward(self, dream_env, track):
        robot_speed = copy.deepcopy(dream_env.robot_speed)
        cur_pos = copy.deepcopy(dream_env.cur_pos)
        cur_angle = copy.deepcopy(dream_env.cur_angle)
        state = copy.deepcopy(dream_env.state),
        last_action = copy.deepcopy(dream_env.last_action),
        wheelVels = copy.deepcopy(dream_env.wheelVels),
        delta_time = copy.deepcopy(dream_env.delta_time)

        # predict 3 steps ahead
        rollout = self.predict_rollout_head(3, dream_env)

        dream_env.set_env_params(robot_speed, cur_pos, cur_angle, state[0], last_action, wheelVels, delta_time)

        cur_tile = dream_env.get_tile()[1]
        tile_kind = dream_env._get_tile(cur_tile[0], cur_tile[1])['kind']

        tree_x = []
        tree_y = []
        min_loss = MIN
        best_node = None
        for node in rollout.nodes:
            # if it's it's not the root
            if node > 1:
                position = rollout.nodes[node]['position']
                angle = rollout.nodes[node]['angle']
                aug_rew = 1.

                # for debugging purpose
                tree_x.append(position[0])
                tree_y.append(position[2])

                try:
                    lane = self.env.get_lane_pos2(position, angle)
                except NotInLane:
                    break

                # LOSS-1: distance to the center of the lane
                """
                dist is:
                    + : left
                    - : right
                """
                dist_lane = np.abs(lane.dist)

                # LOSS-2: the direction of the agent in relation to the (direction of the) lane
                """
                dot_dir is:
                    +1: perfectly aligned looking ahead 
                    -1: perfectly aligned looking back
                     0: angle_deg=90  
                """
                # dot_dir = abs(lane.dot_dir)
                angle_lane = np.abs(lane.angle_deg)

                # prefer nodes with FULL_SPEED when in straight tile
                if rollout.nodes[node]["action_sequence"][0] == 0:
                    if not tile_kind.startswith("curve"):
                        aug_rew = 0.8

                # the node that has the min LOSS = (LOSS-1 + LOSS-2) is the best node
                loss = (dist_lane + angle_lane) * aug_rew
                if loss < min_loss:
                    min_loss = loss
                    best_node = node

        # best_node_pos = rollout.nodes[best_node]['position']
        # plt.scatter(dream_env.cur_pos[0], dream_env.cur_pos[2], color='green', s=1)
        # plt.scatter(tree_x, tree_y, color='blue', s=1)
        # plt.scatter(best_node_pos[0], best_node_pos[2], color='red', s=1)
        # plt.scatter(track[0], track[1], color='black', s=1)
        # plt.show()

        if best_node is not None:
            action_seq = rollout.nodes[best_node]['action_sequence']
            next_action = self.actions[action_seq[0]]
        else:
            action = np.random.randint(0, self.actions.shape[0])
            next_action = self.actions[action]

        return next_action

    def dream_forward_rew(self, dream_env, track):
        robot_speed = copy.deepcopy(dream_env.robot_speed)
        cur_pos = copy.deepcopy(dream_env.cur_pos)
        cur_angle = copy.deepcopy(dream_env.cur_angle)
        state = copy.deepcopy(dream_env.state),
        last_action = copy.deepcopy(dream_env.last_action),
        wheelVels = copy.deepcopy(dream_env.wheelVels),
        delta_time = copy.deepcopy(dream_env.delta_time)

        # predict 3 steps ahead
        rollout = self.predict_rollout_head(3, dream_env)

        dream_env.set_env_params(robot_speed, cur_pos, cur_angle, state[0], last_action, wheelVels[0], delta_time)

        cur_tile = dream_env.get_tile()[1]
        tile_kind = dream_env._get_tile(cur_tile[0], cur_tile[1])['kind']
        print(tile_kind)

        tree_x = []
        tree_y = []
        max_reward = MAX
        best_node = None
        for node in rollout.nodes:
            # if it's not the root
            if node > 1:
                position = rollout.nodes[node]['position']
                angle = rollout.nodes[node]['angle']
                speed = rollout.nodes[node]['speed']
                wheelVels = rollout.nodes[node]['wheelVels']
                aug_rew = 1.

                # for debugging purpose
                tree_x.append(position[0])
                tree_y.append(position[2])


                # prefer nodes with FULL_SPEED sequence
                if rollout.nodes[node]['action_sequence'][0] == 0:
                    if not tile_kind.startswith("curve_"):
                        aug_rew = 1.1
                    else:
                        aug_rew = 0.5

                # the node that has the min LOSS = (LOSS-1 + LOSS-2) is the best node
                reward = dream_env.compute_reward(position, angle, speed, wheelVels) * aug_rew

                if reward > max_reward:
                    max_reward = reward
                    best_node = node

        # best_node_pos = rollout.nodes[best_node]['position']
        # plt.scatter(dream_env.cur_pos[0], dream_env.cur_pos[2], color='green', s=1)
        # plt.scatter(tree_x, tree_y, color='blue', s=1)
        # plt.scatter(best_node_pos[0], best_node_pos[2], color='red', s=1)
        # plt.scatter(track[0], track[1], color='black', s=1)
        # plt.show()

        if best_node is not None:
            action_seq = rollout.nodes[best_node]['action_sequence']
            next_action = self.actions[action_seq[0]]
        else:
            action = np.random.randint(0, self.actions.shape[0])
            next_action = self.actions[action]

        return next_action
