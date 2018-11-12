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
def create_input_placeholders(obs_dim, long_mem_size, n_acts, scope):
    with tf.variable_scope(scope):
        obs_ph = tf.placeholder(shape=(None, *obs_dim), dtype=tf.float32, name="obs_ph_"+scope)
        # mem_acts_ph = tf.placeholder(shape=(None, long_mem_size, n_acts, 1), dtype=tf.float32, name="mem_acts_ph_"+scope)
        # mem_rews_ph = tf.placeholder(shape=(None, long_mem_size, 1, 1), dtype=tf.float32, name="mem_rews_ph_"+scope)
    return {'obs_ph': obs_ph,
            # 'mem_acts_ph': mem_acts_ph,
            # 'mem_rews_ph': mem_rews_ph,
            }

# make output placeholders
def create_output_placeholders(n_acts, scope):
    with tf.variable_scope(scope):
        act_ph = tf.placeholder(shape=(None, n_acts), dtype=tf.int32, name="action_ph_"+scope)
        rew_ph = tf.placeholder(shape=(None,), dtype=tf.float32, name="reward_ph_"+scope)
        future_rew_avg_ph = tf.placeholder(shape=(None,), dtype=tf.float32, name="future_rew_avg_ph_"+scope)
        done_ph = tf.placeholder(shape=(None,), dtype=tf.float32, name="done_ph_"+scope)
        priority_ph = tf.placeholder(shape=(None,), dtype=tf.float32, name="priority_ph_"+scope)
    return {'act_ph': act_ph,
            'rew_ph': rew_ph,
            'future_rew_avg_ph': future_rew_avg_ph,
            'done_ph': done_ph,
            'priority_ph': priority_ph}

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
        cur_input = create_input_placeholders(obs_dim, long_mem_size, n_acts, 'cur_input')
        next_input = create_input_placeholders(obs_dim, long_mem_size, n_acts, 'next_input')

        # create placeholders for the output data for the current timestep
        cur_output = create_output_placeholders(n_acts, 'cur_out')

        # Note: This code relies on only one action happening at once.
        #       It would need rewrites to handle multiple simultaneous actions.
        with tf.variable_scope('processed_output'):
            if future_rew_avg:
                # print(sess.run(cur_output['rew_ph']))
                # print(sess.run(cur_output['future_rew_avg_ph']))
                act_ph = [cur_output['act_ph'], cur_output['act_ph']]
                rew_ph = [cur_output['rew_ph'], cur_output['future_rew_avg_ph']]
                # rew_ph = [cur_output['rew_ph'], cur_output['rew_ph']]
            else:
                act_ph = cur_output['act_ph']
                rew_ph = cur_output['rew_ph']


        # calculate the q value for the chosen action
        q_vals_main_cur = get_model(cur_input['obs_ph'], n_acts, 'main')
        q_a = tf.reduce_max(tf.cast(act_ph, dtype=tf.float32) * q_vals_main_cur, axis=-1)

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
        q_target = rew_ph + q_target

        # create the loss function
        td_error = q_a - tf.stop_gradient(q_target)
        adjusted_square_error = U.huber_loss(cur_output['priority_ph'] * td_error)
        loss = tf.reduce_mean(adjusted_square_error)

        # make target update operations
        main_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='main')
        target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target')
        if use_hold_vars:
            q_vals_hold = get_model(cur_input['obs_ph'], n_acts, 'hold')
            hold_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='hold')
            assign_ops_hold = [tf.assign(hold_var, main_var) for hold_var, main_var in zip(hold_vars, main_vars)]
            assign_ops_target = [tf.assign(target_var, hold_var) for target_var, hold_var in zip(target_vars, hold_vars)]
            target_hold_update_op = tf.group(*assign_ops_hold)
        else:
            assign_ops_target = [tf.assign(target_var, main_var) for target_var, main_var in zip(target_vars, main_vars)]
            target_hold_update_op = None

        target_update_op = tf.group(*assign_ops_target)
        def update_target_hold():
            if target_hold_update_op is not None:
                sess.run(target_hold_update_op)
        def update_target():
            sess.run(target_update_op)

        # make train function
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        gradients = list(gradients)
        if grad_val_clipping:
            for i, grad in enumerate(gradients):
                if grad is not None:
                    gradients[i] = tf.clip_by_value(grad, -grad_val_clipping, grad_val_clipping)
        if grad_norm_clipping:
            gradients, global_norm = tf.clip_by_global_norm(gradients, grad_norm_clipping)
        train_op = optimizer.apply_gradients(zip(gradients, variables))
        def train(batch):
            future_rews = batch['cur_mem']['future_rews']
            future_rew_avg = np.average(future_rews, axis=-1)
            feed_dict = {cur_input['obs_ph']: batch['cur_obs'],
                         next_input['obs_ph']: batch['next_obs'],
                         cur_output['act_ph']: batch['acts'],
                         cur_output['rew_ph']: batch['rews'],
                         cur_output['future_rew_avg_ph']: future_rew_avg,
                         cur_output['done_ph']: batch['done'],
                         cur_output['priority_ph']: batch['priority'],
                         # cur_input['mem_acts_ph']: np.expand_dims(batch['cur_mem']['acts'], 3),
                         # cur_input['mem_rews_ph']: np.expand_dims(np.expand_dims(batch['cur_mem']['rews'], 2), 3),
                         # next_input['mem_acts_ph']: np.expand_dims(batch['next_mem']['acts'], 3),
                         # next_input['mem_rews_ph']: np.expand_dims(np.expand_dims(batch['next_mem']['rews'], 2), 3)
                         }
            sess.run(train_op, feed_dict=feed_dict)
            # print(sess.run(global_norm, feed_dict=feed_dict))


        self.train = train
        self.update_target = update_target
        self.update_target_hold = update_target_hold
        self.cur_input = cur_input
        self.q_vals = tf.concat(q_vals_main_cur, 0)

        self.save = functools.partial(U.save_variables, sess=sess)
        self.load = functools.partial(U.load_variables, sess=sess)

    # Initialize the parameters and copy them to the target network.
    def initialize_training(self):
        U.initialize()
        self.update_target()

    # TODO: Make sure that act has the right shape/etc
    def choose_action(self, epsilon, cur_mem, obs, rc, env):
        n_acts = env.action_space.n
        obs_ph = self.cur_input['obs_ph']
        # mem_acts_ph = self.cur_input['mem_acts_ph']
        # mem_rews_ph = self.cur_input['mem_rews_ph']
        q_vals = self.q_vals

        # If repeat count > 0, keep using the same actions as last time.
        if rc > 0:
            rc -= 1
            act = cur_mem['acts'][-1]
            act_env = np.argmax(act)
            # print("Repeated act of", act_env)
            return (act, act_env, rc)

        # Choose whether to act randomly or to act according to the current q
        if np.random.rand() < epsilon:
            act_env = env.action_space.sample()
            max_hold_length = 2
            rc = np.random.randint(max_hold_length)
            # print("Random act of", act_env)
        else:
            obs_array = np.asarray(obs)
            reshaped_obs_array = obs_array.reshape(1, *obs_array.shape) if len(obs_array.shape) > 2 else obs_array
            cur_q = self.sess.run(q_vals,
                                  feed_dict={obs_ph: reshaped_obs_array}) #,
                                             # mem_acts_ph: cur_mem["acts"].reshape(1, *cur_mem["acts"].shape, 1),
                                             # mem_rews_ph: cur_mem["rews"].reshape(1, *cur_mem["rews"].shape, 1, 1)})
            # print(cur_q[0])
            # act = np.array([int(possible_act > 0) for possible_act in cur_q[0]])
            # print(cur_q)
            act_env = np.argmax(cur_q, axis=-1)[0] % n_acts
            # print(act_env)
            # act_env = np.random.choice(n_acts, 1, p=cur_q[0])
            rc = 0
            # print("Chose act of ", act_env)
        act = np.array([float(i == act_env) for i in range(n_acts)])

        return (act, act_env, rc)


# TODO: adjust the model's layers more
def get_model(obs_in, #mem_in_acts, mem_in_rews, 
              n_acts, scope,
              # colors=False,
              # convs=[],
              # mem_convs=[],
              # convs=[[8, 1, 24, 2], [4, 1, 12, 2], [4, 1, 6, 2]],
              # mem_convs=[[3, 6, 3, 2], [3, 1, 4, 1]],
              # colors=[24, 1],
              # # convs=[[8, 1, 3, 12], [4, 1, 12, 3], [4, 1, 3, 1]],
              # convs=[[16, 4, 25, 12], [4, 1, 12, 3], [4, 1, 3, 1]],
              hiddens=[24, 24],
              dueling=True,
              future_rew_avg=False,
              layer_norm=True):
    # print("n_acts = ", n_acts)
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        obs_out = obs_in

        # # First, learn the useful colors
        # with tf.variable_scope("color_net"):
        #     if colors:
        #         (num_colors, num_color_mods) = colors
        #         obs_out = conv2d(obs_in,
        #                          filter_size=1,
        #                          num_filters=num_colors,
        #                          stride=1,
        #                          activation=tf.nn.softmax)
        #         if num_color_mods > 0:
        #             obs_out2 = conv2d(obs_in,
        #                              filter_size=1,
        #                              num_filters=num_color_mods,
        #                              stride=1,
        #                              activation=tf.nn.sigmoid)
        #             obs_out = tf.concat([obs_out, obs_out2], -1)
        # print(obs_out.shape)

        # Next, apply a series of convolutional layers.
        conv_out = obs_out
        # with tf.variable_scope("convnet"):
        #     for filter_size, stride, num_filters, pool_size in convs:
        #         obs_out = conv2d(obs_out,
        #                          filter_size=filter_size,
        #                          num_filters=num_filters,
        #                          stride=stride,
        #                          activation=tf.nn.relu)
        #         print(obs_out.shape)
        #         if pool_size > 1:
        #             obs_out = max_pool(obs_out, pool_size)
        #         print(obs_out.shape)
        #     conv1_out = tf.contrib.layers.flatten(obs_out)
        #     print(conv1_out.shape)

        # # Next, analyze past actions and rewards
        # mem_out_acts = mem_in_acts
        # mem_out_rews = mem_in_rews
        # print("mem_in_acts.shape =", mem_in_acts.shape)
        # print("mem_in_rews.shape =", mem_in_rews.shape)
        # with tf.variable_scope("mem_convnet"):
        #     for filter_size1, filter_size2, num_filters, pool_size in mem_convs:
        #         mem_out_acts = conv2d(mem_out_acts,
        #                               filter_size=filter_size1,
        #                               filter_size2=filter_size2,
        #                               num_filters=num_filters,
        #                               stride=1,
        #                               activation=tf.nn.relu)
        #         mem_out_acts = max_pool(mem_out_acts, pool_size, 1)
        #         mem_out_rews = conv2d(mem_out_rews,
        #                               filter_size=filter_size1,
        #                               filter_size2=1,
        #                               num_filters=num_filters,
        #                               stride=1,
        #                               activation=tf.nn.relu)
        #         mem_out_rews = max_pool(mem_out_rews, pool_size, 1)
        #         print("mem_out_acts.shape =", mem_out_acts.shape)
        #         print("mem_out_rews.shape =", mem_out_rews.shape)
        #     mem_out = fc(tf.concat([mem_out_acts, mem_out_rews], 1), 30, activation=tf.nn.relu)
        #     print("mem_out.shape =", mem_out.shape)
        #     # mem_out = tf.contrib.layers.flatten(tf.concat([mem_out_acts, mem_out_rews], 1))
        # conv_out = tf.concat([conv1_out, mem_out], 1)


        with tf.variable_scope("action_value"):
            action_out = conv_out
            for hidden in hiddens:
                action_out = tf.contrib.layers.fully_connected(action_out, num_outputs=hidden, activation_fn=None)
                if layer_norm:
                    action_out = tf.contrib.layers.layer_norm(action_out, center=True, scale=True)
                action_out = tf.nn.relu(action_out)
            action_scores = tf.contrib.layers.fully_connected(action_out, num_outputs=n_acts, activation_fn=None)

        if dueling:
            with tf.variable_scope("state_value"):
                state_out = conv_out
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

        if future_rew_avg:
            with tf.variable_scope("future_rewards"):
                future_rew_in = tf.concat([q_out, conv_out], 1)
                with tf.variable_scope("action_value"):
                    action_out = future_rew_in
                    for hidden in hiddens:
                        action_out = tf.contrib.layers.fully_connected(action_out, num_outputs=hidden, activation_fn=None)
                        if layer_norm:
                            action_out = tf.contrib.layers.layer_norm(action_out, center=True, scale=True)
                        action_out = tf.nn.relu(action_out)
                    action_scores = tf.contrib.layers.fully_connected(action_out, num_outputs=n_acts, activation_fn=None)

                if dueling:
                    with tf.variable_scope("state_value"):
                        state_out = future_rew_in
                        for hidden in hiddens:
                            state_out = tf.contrib.layers.fully_connected(state_out, num_outputs=hidden, activation_fn=None)
                            if layer_norm:
                                state_out = tf.contrib.layers.layer_norm(state_out, center=True, scale=True)
                            state_out = tf.nn.relu(state_out)
                        state_score = tf.contrib.layers.fully_connected(state_out, num_outputs=1, activation_fn=None)
                    action_scores_mean = tf.reduce_mean(action_scores, 1)
                    action_scores_centered = action_scores - tf.expand_dims(action_scores_mean, 1)
                    q_out_2 = state_score + action_scores_centered
                else:
                    q_out_2 = action_scores
                q_out = tf.concat([[q_out], [q_out_2]], 0)            

        return q_out
