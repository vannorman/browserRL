import time

import numpy as np

from replay_buffer import ReplayBuffer
from models import Model

print("hello")


def createReplayBuffer(obs_dim, n_acts, buffer_size=50000):
    return ReplayBuffer(obs_dim, n_acts, buffer_size)


class DQN:
    def __init__(self,
                 obs_dim,
                 n_acts,
                 lr=1e-4,
                 gamma=0.99,
                 load_path=None,
                 seed=None):
        if seed is None:
            seed = int(time.time()*1000000) - int(time.time())*1000000
        self.train_count = 0

        self.model = Model(obs_dim, n_acts, seed, lr, gamma)
        self.replay_buffer = createReplayBuffer(obs_dim, n_acts)

        if load_path is not None:
            self.model.load(load_path)
            print("Loaded model from " + load_path + ".")

        self.model.initialize_training()
        print("Initialized DQN")


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
            updated_model = self.train(batch_size, target_update_interval)
        
        # https://js.tensorflow.org/tutorials/import-saved-model.html
        
        
        return updated_model

fake_data_batch_size = 100
fake_data_obs_dim = 12
fake_data_n_acts = 4
my_dqn = DQN([fake_data_obs_dim], fake_data_n_acts)

fake_data = [{'obs': np.random.rand(fake_data_obs_dim),
              'act': np.random.randint(2, size=fake_data_n_acts),
              'rew': np.random.rand(1),
              'done': np.random.randint(2)
            } for i in range(fake_data_batch_size)]
updated_model = my_dqn.store_and_train(fake_data)

print(updated_model)

