import time

import numpy as np

from rl_algs.deepq.replay_buffer import ReplayBuffer
from rl_algs.deepq.models import Model


def createReplayBuffer(obs_dim, n_acts, buffer_size=50000):
    return ReplayBuffer(obs_dim, n_acts, buffer_size)


class DQN:
    def __init__(self,
                 obs_dim,
                 n_acts,
                 lr=1e-4,
                 gamma=0.99,
                 final_epsilon=0.02,
                 load_path=None,
                 seed=None):
        if seed is None:
            seed = int(time.time()*1000000) - int(time.time())*1000000
        self.epsilon_schedule = getEpsilonSchedule(total_timesteps, final_epsilon)
        self.train_count = 0

        self.model = Model(obs_dim, n_acts, seed, lr, gamma)
        self.replay_buffer = createReplayBuffer(obs_dim, n_acts)

        if load_path is not None:
            self.model.load(load_path)
            print("Loaded model from " + load_path + ".")

        self.model.initialize_training()


    def saveModel(self, model_path=None):
        if model_path is not None:
            self.model.save(model_path)
            print("Saved model to " + model_path + ".")


    def accept_data(self, batch):
        self.replay_buffer.store_batch(batch)


    def train(self,
              batch_size=32,
              target_update_interval=500):
        train_count = self.train_count

        batch = self.replay_buffer.sample(batch_size)
        self.model.train(batch)
        # print('Finished training session')

        if train_count % target_update_interval == 0:
            self.model.update_target()
            print('Updated target')

        return self.model #TODO: What exactly should this return?


    def store_and_train(self,
                        batch,
                        batch_size=32,
                        target_update_interval=500,
                        trainings_per_request=1,
                        old_model=None):
        self.accept_data(batch)
        for i in range(trainings_per_request):
            updated_model = train(batch_size, target_update_interval)
        return updated_model

