import gym
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import numpy as np
import matplotlib.pyplot as plt
import os
from utils_legacy import mlp_actor_critic
from utils import placeholder, placeholders, get_vars, count_vars, ReplayBuffer, sample_param

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def ddpg_online(env, env_idx, policy_file=None, init_policy=None, start=0, end=0,
         gamma=1.0, epochs=5, pi_lr=1e-3, q_lr=1e-3,
         hidden_sizes=(256, 256), activation=tf.nn.relu,
         max_ep_len=4000, save_freq=5, steps_per_epoch=4000,
         replay_size=int(1e8), polyak=0.995, batch_size=100,
         update_after=200, update_every=200, act_noise=0.001,
         rand_act_ratio=0, warmstart_steps=0):

    print(">>> steps_per_epoch =", steps_per_epoch, "epochs =", epochs)

    tf.compat.v1.reset_default_graph()
    seed = 0
    tf.compat.v1.set_random_seed(seed)
    np.random.seed(seed)

    start_steps = int(steps_per_epoch * epochs * rand_act_ratio)

    obs_dim = env.observation_space.shape[0]
    act_dim = 1

    act_limit_h = env.action_space.high[0]
    act_limit_l = env.action_space.low[0]

    x_ph, a_ph, x2_ph, r_ph, d_ph = placeholders(obs_dim, act_dim, obs_dim, None, None)

    with tf.compat.v1.variable_scope('main'):
        pi, q, q_pi = mlp_actor_critic(
            x_ph, a_ph,
            hidden_sizes=hidden_sizes,
            activation=activation,
            output_activation=tf.nn.tanh,
            action_space=env.action_space
        )

    with tf.compat.v1.variable_scope('target'):
        pi_targ, _, q_pi_targ = mlp_actor_critic(
            x2_ph, a_ph,
            hidden_sizes=hidden_sizes,
            activation=activation,
            output_activation=tf.nn.tanh,
            action_space=env.action_space
        )

    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    backup = tf.stop_gradient(r_ph + gamma * (1 - d_ph) * q_pi_targ)

    pi_loss = -tf.reduce_mean(q_pi)
    q_loss = tf.reduce_mean((q - backup) ** 2)

    pi_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=pi_lr)
    q_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=q_lr)
    train_pi_op = pi_optimizer.minimize(pi_loss, var_list=get_vars('main/pi'))
    train_q_op = q_optimizer.minimize(q_loss, var_list=get_vars('main/q'))

    target_update = tf.group([
        tf.compat.v1.assign(v_targ, polyak * v_targ + (1 - polyak) * v_main)
        for v_main, v_targ in zip(get_vars('main'), get_vars('target'))
    ])

    target_init = tf.group([
        tf.compat.v1.assign(v_targ, v_main)
        for v_main, v_targ in zip(get_vars('main'), get_vars('target'))
    ])

    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())
    sess.run(target_init)

    def get_action(o, noise_scale, pi=pi):
        a = sess.run(pi, feed_dict={x_ph: o.reshape(1, -1)})[0]
        a += noise_scale * np.random.randn(act_dim)
        return np.clip(a, act_limit_l, act_limit_h)

    def get_init_action(o, noise_scale=0.0):
        a = init_policy(o)
        a = np.asarray(a, dtype=np.float32).reshape(act_dim)
        if noise_scale > 0:
            a += noise_scale * np.random.randn(act_dim)
        return np.clip(a, act_limit_l, act_limit_h)

    total_steps = steps_per_epoch * epochs

    saver = tf.compat.v1.train.Saver()
    if policy_file is not None:
        saver.restore(sess, policy_file)
        print(">>> restored TF policy from:", policy_file)
    else:
        print(">>> no TF checkpoint restored; using random DDPG init")

    obs_list = []
    reward_list = []
    action_list = []
    energy_list = [0.0]
    penalty_list = [0.0]
    temp_metric_list = [0.0]
    lb_list = []
    ub_list = []

    o = env.reset()
    ep_ret = 0.0
    ep_len = 0
    ep_ret_print = 0.0
    episode_count = 0

    for t in range(total_steps):

        if t == 0:
            print(">>> loop entered, total_steps =", total_steps)

        if t <= start_steps:
            a = env.action_space.sample()
        elif init_policy is not None and t < warmstart_steps:
            a = get_init_action(o, noise_scale=act_noise)
        else:
            a = get_action(o, act_noise)

        obs_list.append(o.copy())

        o2, r, d, info = env.step(a)

        ep_ret += float(r)
        ep_len += 1

        reward_list.append(r)
        action_list.append(float(a[0]))

        energy_list.append(energy_list[-1] + float(info['Energy']))
        penalty_list.append(penalty_list[-1] + float(info['Penalty']))
        temp_metric_list.append(temp_metric_list[-1] + float(info['Exceedance']))
        lb_list.append(info['lb'])
        ub_list.append(info['ub'])

        replay_buffer.store(o, a, r, o2, d)
        o = o2

        if d:
            episode_count += 1
            ep_ret_print = ep_ret
            print(
                "Episode", episode_count,
                "done at global_t =", t,
                "| ep_len =", ep_len,
                "| ep_ret =", ep_ret
            )
            o = env.reset()
            ep_ret, ep_len = 0.0, 0
            continue

        if ep_len >= max_ep_len:
            episode_count += 1
            ep_ret_print = ep_ret
            print(
                "Episode", episode_count,
                "cut by max_ep_len at global_t =", t,
                "| ep_len =", ep_len,
                "| ep_ret =", ep_ret
            )
            o = env.reset()
            ep_ret, ep_len = 0.0, 0

        if t >= update_after and (t % update_every == 0):
            for _ in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                feed_dict = {
                    x_ph: batch['obs1'],
                    x2_ph: batch['obs2'],
                    a_ph: batch['acts'],
                    r_ph: batch['rews'],
                    d_ph: batch['done'],
                }
                sess.run([q_loss, q, train_q_op], feed_dict)
                sess.run([pi_loss, train_pi_op, target_update], feed_dict)

        if (t + 1) % steps_per_epoch == 0:
            epoch = (t + 1) // steps_per_epoch
            print('epoch:', epoch, 'last_ep_ret:', ep_ret_print, '| episodes_so_far:', episode_count)

            if (epoch % save_freq == 0) or (epoch == epochs):
                print(">>> saving at epoch", epoch)
                save_path = saver.save(sess, "./model/online/saved_model_%d.ckpt" % env_idx)
                print("Model saved in path: %s" % save_path)

    save_path = saver.save(sess, "./model/online/saved_model_%d.ckpt" % env_idx)
    print("Model saved in path:", save_path)
    sess.close()

    low = np.array([10.0, 18.0, 21.0, -40.0, 0., 50., 0])
    high = np.array([35.0, 27.0, 23.0, 40.0, 1100., 180., 23])

    obs_arr = np.array(obs_list)
    T_air = obs_arr[:, 1] * (high[1] - low[1]) + low[1]
    T_out = obs_arr[:, 3] * (high[3] - low[3]) + low[3]
    Q_SG = obs_arr[:, 4] * (high[4] - low[4]) + low[4]

    time = np.linspace(start, end, num=len(T_air), endpoint=False)

    return (
        T_air,
        time,
        T_out,
        Q_SG,
        np.array(action_list),
        np.array(energy_list[1:]),
        np.array(penalty_list[1:]),
        np.array(temp_metric_list[1:]),
        lb_list,
        ub_list
    )
