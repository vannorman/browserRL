import time

import numpy as np

from rl_algs.deepq.replay_buffer import ReplayBuffer
from rl_algs.deepq.models import Model


def getEpsilonSchedule(total_timesteps,
        final_epsilon,
        entirely_random_steps=15000,
        somewhat_random_steps=10000,
        decaying_random_steps=10000,
        intermediate_epsilon=0.1):

    def getEpsilon(t):
        if t < entirely_random_steps:
            epsilon = 1
        elif t < entirely_random_steps + somewhat_random_steps:
            decay = (1 - intermediate_epsilon)*min(1, (t - entirely_random_steps)/somewhat_random_steps)
            epsilon = 1 - decay
        else:
            epsilon = final_epsilon
        return epsilon
    return [getEpsilon(t) for t in range(total_timesteps)]


def createReplayBuffer(obs_dim, n_acts, buffer_size=50000):
    return ReplayBuffer(obs_dim, n_acts, buffer_size, long_mem_size)


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
        epsilon_schedule = self.epsilon_schedule
        t = self.train_count

        batch = self.replay_buffer.sample(batch_size)
        self.model.train(batch)
        # print('Finished training session')

        if t % target_update_interval == 0:
            # Update the target network
            if t % target_update_interval == 0:
                self.model.update_target_hold()

            if t % target_update_interval == 0:
                self.model.update_target()
                print('Updated target')


        return self.model #TODO: What exactly should this return?

    def store_and_train(self, data):
        batch_size = data['batch_size'] or 32
        target_update_interval = data['target_update_interval'] or 500
        trainings_per_request = data['trainings_per_request'] or 1
        checkp
        old_model = data['model']
        self.accept_data(data['batch'])
        for i in range(trainings_per_request):
            updated_model = train(batch_size, target_update_interval)
        return updated_model

