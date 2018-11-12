import functools
import numpy as np
import tensorflow as tf

import baselines.common.tf_util as U
from baselines.common import set_global_seeds

layers = tf.layers


def layer_out(z, layer_norm=True, activation=tf.nn.tanh, dropout_rate=False):
    if dropout_rate:
        z = tf.nn.dropout(z, keep_prob=(1-dropout_rate))
    if layer_norm:
        z = tf.contrib.layers.layer_norm(z, center=True, scale=True)
    return activation(z)


def fc(x, num_out, layer_norm=True, activation=tf.nn.tanh, dropout_rate=False):
    x_flat = tf.contrib.layers.flatten(x) # TODO: Does doing this every time cause any issues?
    W = weight_variable([x_flat.shape[1].value, num_out])
    b = bias_variable([num_out])
    z = tf.matmul(x_flat, W) + b
    return layer_out(z, layer_norm, activation, dropout_rate)


def max_pool(x, size, size2=None):
    """max_pool downsamples a feature map by size."""
    if size2 is None:
        size2 = size
    return tf.nn.max_pool(x, ksize=[1, size, size2, 1],
                          strides=[1, size, size2, 1], padding='VALID')


def conv2d(x, filter_size, num_filters, stride=1, layer_norm=True, activation=tf.nn.relu, filter_size2=False):
    channels_in = x.shape[3].value
    filter_size2 = filter_size2 or filter_size
    W = weight_variable([filter_size, filter_size2, channels_in, num_filters])
    b = bias_variable([num_filters])
    z = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID') + b
    return layer_out(z, layer_norm, activation)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# make input placeholders
def create_input_placeholders(obs_dim, scope):
    with tf.variable_scope(scope):
        obs_ph = tf.placeholder(shape=(None, *obs_dim), dtype=tf.float32, name="obs_ph_"+scope)
    return {'obs_ph': obs_ph}


# make output placeholders
def create_output_placeholders(n_acts, scope):
    with tf.variable_scope(scope):
        act_ph = tf.placeholder(shape=(None, n_acts), dtype=tf.int32, name="action_ph_"+scope)
        rew_ph = tf.placeholder(shape=(None,), dtype=tf.float32, name="reward_ph_"+scope)
        done_ph = tf.placeholder(shape=(None,), dtype=tf.float32, name="done_ph_"+scope)
    return {'act_ph': act_ph,
            'rew_ph': rew_ph,
            'done_ph': done_ph}


class Model:
    def __init__(self,
                 obs_dim,
                 n_acts,
                 seed,
                 lr,
                 gamma,
                 double_q=True,
                 grad_val_clipping=None,
                 grad_norm_clipping=None):
                 # grad_val_clipping=0.5,
                 # grad_norm_clipping=5.0):

        sess = U.get_session()
        self.sess = sess
        set_global_seeds(seed)

        # create placeholders for the input data for the current and next timesteps
        cur_input = create_input_placeholders(obs_dim, 'cur_input')
        next_input = create_input_placeholders(obs_dim, 'next_input')

        # create placeholders for the output data for the current timestep
        cur_output = create_output_placeholders(n_acts, 'cur_out')

        # calculate the q value for the chosen action
        q_vals_main_cur = get_model(cur_input['obs_ph'], n_acts, 'main')
        q_a = tf.reduce_max(tf.cast(cur_output['act_ph'], dtype=tf.float32) * q_vals_main_cur, axis=-1)

        # calculate the q value for the target network
        q_vals_target_next = get_model(next_input['obs_ph'], n_acts, 'target')
        if double_q:
            q_vals_main_next = get_model(next_input['obs_ph'], n_acts, 'main')
            next_act_main = tf.argmax(q_vals_main_next, axis=-1)
            q_vals_target_next_best = tf.reduce_max(q_vals_target_next * tf.one_hot(next_act_main, n_acts), axis=-1)
        else:
            q_vals_target_next_best = tf.reduce_max(q_vals_target_next, axis=-1)

        done_mask = 1 - cur_output['done_ph']
        q_target = done_mask * gamma * q_vals_target_next_best
        q_target = cur_output['rew_ph'] + q_target

        # create the loss function
        td_error = q_a - tf.stop_gradient(q_target)
        adjusted_square_error = U.huber_loss(td_error)
        loss = tf.reduce_mean(adjusted_square_error)

        # make target update operation
        main_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='main')
        target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target')
        assign_ops_target = [tf.assign(target_var, main_var) for target_var, main_var in zip(target_vars, main_vars)]
        target_update_op = tf.group(*assign_ops_target)
        def update_target():
            sess.run(target_update_op)

        # make train function
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        gradients = list(gradients)
        # if grad_val_clipping:
        #     for i, grad in enumerate(gradients):
        #         if grad is not None:
        #             gradients[i] = tf.clip_by_value(grad, -grad_val_clipping, grad_val_clipping)
        # if grad_norm_clipping:
        #     gradients, global_norm = tf.clip_by_global_norm(gradients, grad_norm_clipping)
        train_op = optimizer.apply_gradients(zip(gradients, variables))
        def train(batch):
            feed_dict = {cur_input['obs_ph']: batch['cur_obs'],
                         next_input['obs_ph']: batch['next_obs'],
                         cur_output['act_ph']: batch['acts'],
                         cur_output['rew_ph']: batch['rews'],
                         cur_output['done_ph']: batch['done'],
                         }
            sess.run(train_op, feed_dict=feed_dict)

        self.train = train
        self.update_target = update_target

        self.save = functools.partial(U.save_variables, sess=sess)
        self.load = functools.partial(U.load_variables, sess=sess)
        print("Initialized Model")

    # Initialize the parameters and copy them to the target network.
    def initialize_training(self):
        U.initialize()
        self.update_target()


# TODO: adjust the model's layers more
def get_model(obs_in,
              n_acts,
              scope,
              hiddens=[24, 24],
              dueling=True,
              layer_norm=True):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        obs_out = obs_in

        with tf.variable_scope("action_value"):
            action_out = obs_out
            for hidden in hiddens:
                action_out = tf.contrib.layers.fully_connected(action_out, num_outputs=hidden, activation_fn=None)
                if layer_norm:
                    action_out = tf.contrib.layers.layer_norm(action_out, center=True, scale=True)
                action_out = tf.nn.relu(action_out)
            action_scores = tf.contrib.layers.fully_connected(action_out, num_outputs=n_acts, activation_fn=None)

        if dueling:
            with tf.variable_scope("state_value"):
                state_out = obs_out
                for hidden in hiddens:
                    state_out = tf.contrib.layers.fully_connected(state_out, num_outputs=hidden, activation_fn=None)
                    if layer_norm:
                        state_out = tf.contrib.layers.layer_norm(state_out, center=True, scale=True)
                    state_out = tf.nn.relu(state_out)
                state_score = tf.contrib.layers.fully_connected(state_out, num_outputs=1, activation_fn=None)
            action_scores_mean = tf.reduce_mean(action_scores, 1)
            action_scores_centered = action_scores - tf.expand_dims(action_scores_mean, 1)
            q_out = state_score + action_scores_centered
        else:
            q_out = action_scores

        return q_out
