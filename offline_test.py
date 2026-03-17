import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import numpy as np
import os
import gym
from env import ContinuousBuildingControlEnvironment as BEnv
import pandas as pd

from utils import placeholders, custom_plot
from utils_legacy import mlp_actor_critic

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def test_policy(policy_file, start=6000., end=8000.,
                data_file='weather_data_2013_to_2017_winter_pandas.csv',
                hidden_sizes=(64, 64, 64, 64), activation=tf.nn.relu):
    env = BEnv(data_file, start=start, end=end,
               C_env=3.1996e6,
               C_air=3.5187e5,
               R_rc=0.00706,
               R_oe=0.02707,
               R_er=0.00369,
               lb_set=22.,
               ub_set=24.)

    obs, done = env.reset(), False

    obs_dim = env.observation_space.shape[0]
    act_dim = 1

    # Inputs to computation graph
    x_ph, a_ph = placeholders(obs_dim, act_dim)

    with tf.compat.v1.variable_scope('main'):
        pi, q, q_pi = mlp_actor_critic(
            x_ph, a_ph,
            hidden_sizes=hidden_sizes,
            activation=activation,
            output_activation=tf.nn.tanh,
            action_space=env.action_space
        )

    obs_list = []
    reward_list = []
    action_list = []
    u_list = []
    energy_list = [0]
    penalty_list = [0]
    temp_metric_list = [0]
    lb_list = []
    ub_list = []

    with tf.compat.v1.Session() as sess:
        saver = tf.compat.v1.train.Saver()
        saver.restore(sess, policy_file)

        while True:
            if not done:
                obs_list.append(obs.copy())

                action_val = sess.run(pi, feed_dict={x_ph: obs.reshape(1, -1)})[0]
                obs, reward, done, dic = env.step(action_val)

                reward_list.append(reward)
                action_list.append(dic['a_t'] if isinstance(dic['a_t'], float) else dic['a_t'][0])
                u_list.append(dic['u_t'] if isinstance(dic['u_t'], float) else dic['u_t'][0])
                energy_list.append(
                    (dic['Energy'] if isinstance(dic['Energy'], float) else dic['Energy'][0]) + energy_list[-1])
                penalty_list.append(dic['Penalty'] + penalty_list[-1])
                temp_metric_list.append(dic['Exceedance'] + temp_metric_list[-1])
                lb_list.append(dic['lb'])
                ub_list.append(dic['ub'])

            if done:
                break

        env.close()

    # lower bound of observation space
    low = np.array([10.0, 18.0, 21.0, -40.0, 0., 50., 0])
    # upper bound of observation space
    high = np.array([35.0, 27.0, 23.0, 40.0, 1100., 180., 23])

    obs_arr = np.array(obs_list)

    T_air = obs_arr[:, 1] * (high[1] - low[1]) + low[1]
    time = np.linspace(start, end, int((end - start) / 0.5))
    T_out = obs_arr[:, 3] * (high[3] - low[3]) + low[3]
    Q_SG = obs_arr[:, 4] * (high[4] - low[4]) + low[4]

    tf.compat.v1.reset_default_graph()

    return T_air, time, T_out, Q_SG, np.array(action_list), np.array(u_list), np.array(energy_list[1:]), \
        np.array(penalty_list[1:]), np.array(temp_metric_list[1:]), lb_list, ub_list


def main():
    data_file = 'weather_data_2013_to_2017_summer_pandas.csv'

    for idx in range(0, 1):
        # load policy
        policy_file = "./model/online/saved_model_0.ckpt"

        # get data
        T_air, time, T_out, Q_SG, action_list, u_list, energy_list, \
            penalty_list, temp_metric_list, lb_list, ub_list = \
            test_policy(start=17664., end=19872.5, data_file=data_file,
                        hidden_sizes=(64, 64, 64, 64), activation=tf.nn.relu,
                        policy_file=policy_file)

        # plot data
        custom_plot(T_air[2100:2300], time[2100:2300], T_out[2100:2300], Q_SG[2100:2300],
                    action_list[2100:2300], energy_list[2100:2300], penalty_list[2100:2300],
                    lb_list[2100:2300], ub_list[2100:2300], idx, "offline")
        print(idx)

    d = {
        'energy_true': energy_list,
        'penalty_true': penalty_list,
        'exceedance_true': temp_metric_list
    }
    df = pd.DataFrame(data=d)
    df.to_csv('results_true.csv', index=False)

    d_profile = {
        'lb': lb_list,
        'ub': ub_list,
        'Qsg': Q_SG,
        'Tout': T_out,
        'Tair_true': T_air,
        'u_true': u_list
    }
    df_profile = pd.DataFrame(data=d_profile)
    df_profile.to_csv('results_profile.csv', index=False)

    # show outputs
    print("Energy Use in kWh: %d" % energy_list[-1])
    print("# of Hours out of Bounds: %d" % penalty_list[-1])
    print("Temperature Exceedance in degC-hr: %d" % temp_metric_list[-1])


if __name__ == "__main__":
    main()