from concurrent.futures import ThreadPoolExecutor

import os
import pickle


class Logger:
    def __init__(self, env, log_file):
        self.env = env

        self._log_file = open(log_file, 'wb')
        # we log the data in a multithreaded fashion
        self._multithreaded_recording = ThreadPoolExecutor(4)
        self._recording = []

    def log(self, observation, action, reward, done, info):
        x, y, z = self.env.cur_pos
        lp = self.env.get_lane_pos2(self.env.cur_pos, self.env.cur_angle)
        displacement = lp.dist
        angle_deg = lp.angle_deg
        self._recording.append({
            'step': [
                observation,
                (displacement, self.env.cur_angle),  # store <displacement, cur_angles>
                (displacement, angle_deg)  # store <displacement, angle_deg>
            ],
            # this is metadata, you may not use it at all, but it may be helpful for debugging purposes
            'metadata': [
                (x, y, z),  # we store the pose, just in case we need it
                action,
                reward,
                done,
                info
            ]
        })

    def on_episode_done(self):
        self._multithreaded_recording.submit(self._commit)

    def _commit(self):
        # we use pickle to store our data
        pickle.dump(self._recording, self._log_file)
        self._log_file.flush()
        del self._recording[:]

    def close(self):
        self._multithreaded_recording.shutdown()
        self._log_file.close()
        os.chmod(self._log_file.name, 0o444)  # make file read-only after finishing


class Reader:

    def __init__(self, log_file):
        self._log_file = open(log_file, 'rb')

    def read(self):
        end = False
        observations = []
        displacements = []
        angles = []

        while not end:
            try:
                log = pickle.load(self._log_file)
                for entry in log:
                    step = entry['step']
                    observations.append(step[1][0])
                    displacements.append(step[1][1])
            except EOFError:
                end = True

        return observations, (displacements, angles)

    def close(self):
        self._log_file.close()
