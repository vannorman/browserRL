# This file reuses a lot of code from OpenAI baselines/baselines/deepq/replay_buffer.py and
# from a tutorial at https://github.com/jachiam/rl-intro

import numpy as np
import random


class ReplayBuffer:

    def __init__(self, obs_dim, n_acts, size):
        self.obs_buf = np.zeros([size, *obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, n_acts], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr = 0
        self.total_count = 0
        self.size = 0
        self.max_size = size
        self.rew_mean = 0

    def store(self, obs, act, rew, done):
        self.obs_buf[self.ptr] = obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)
        self.total_count += 1
        self.update_stats(rew)

    def get_memory(self, idxs=False):
        if idxs is False:
            idxs = [self.ptr - 1]
        # TODO: These lists include the current timestep in the memory.
        # I think that's good, but consider removing it.

        n_avg = 8
        idxs_future_and_current = [[(idx + j) % self.max_size for j in range(n_avg)] for idx in idxs]

        future_rews = self.rews_buf[idxs_future_and_current] - self.rew_mean
        future_done = self.done_buf[idxs_future_and_current]

        # If one of the items in memory is done (and isn't the current or next timestep),
        # then zero out everything in that timestep and the ones before it.
        # This is not the most efficient way to do this, but it's probably fine unless long_mem_size gets huge.
        done_idxs = np.nonzero(done)
        future_done_idxs = np.nonzero(future_done)

        if len(done_idxs) == 1:
            for i in done_idxs[0]:
                if i + 1 < len(rews):
                    rews[i+1:] = 0
                    acts[i+1:] = np.zeros(acts[i+1:].shape, dtype=np.float32)
                    # obs[i+1:] = np.zeros(obs[i+1:].shape, dtype=np.float32)

        if len(future_done_idxs) == 1:
            for i in future_done_idxs[0]:
                if i + 1 < len(future_rews):
                    future_rews[i+1:] = 0

        return dict(acts=acts,
                    rews=rews,
                    # obs=obs[-self.short_mem_size:],
                    future_rews=future_rews)

    def choose_batch_idxs(self, batch_size, include_most_recent):
        long_mem_size = self.long_mem_size

        idxs = np.random.choice(self.size, batch_size)
        idxs = np.array([(idx + long_mem_size - 1) % self.size for idx in idxs])

        if include_most_recent:
            idxs[-1] = self.ptr - 1

        return idxs

    def get_priority_weights(self, idxs=None):
        return np.ones(idxs.shape).astype(float)

    def sample(self, batch_size=32, include_most_recent=False):
        idxs = self.choose_batch_idxs(batch_size, include_most_recent)
        return dict(cur_obs=self.obs_buf[idxs],
                    next_obs=self.obs_buf[(idxs + 1) % self.max_size],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs] - self.rew_mean,
                    done=self.done_buf[idxs],
                    priority=self.get_priority_weights(idxs),
                    cur_mem=self.get_memory(idxs),
                    next_mem=self.get_memory((idxs + 1) % self.max_size),
                    idxs=idxs)

    def update_stats(self, rew):
        self.rew_mean = (self.rew_mean * (self.total_count - 1) + rew) / self.total_count
