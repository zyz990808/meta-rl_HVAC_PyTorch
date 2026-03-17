from env import ContinuousBuildingControlEnvironment as BEnv
import numpy as np
import os
import pandas as pd
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import torch

from ddpg_online import ddpg_online
from ppo_multi_env_train import ActorCritic

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main():
    data_file = 'weather_data_2013_to_2017_summer_pandas.csv'

    online_start = 6000.
    online_end = 8160.

    env = BEnv(
        data_file,
        start=online_start,
        end=online_end,
        C_env=3.1996e6,
        C_air=3.5187e5,
        R_rc=0.00706,
        R_oe=0.02707,
        R_er=0.00369
    )

    # ===== load PPO model =====
    ppo_path = "model/multi_env/ppo_multi_env_450.pth"

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_low = env.action_space.low
    act_high = env.action_space.high

    ppo_model = ActorCritic(obs_dim, act_dim, act_low, act_high)
    ppo_model.load_state_dict(torch.load(ppo_path, map_location="cpu"))
    ppo_model.eval()

    def init_policy(obs):
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            mean = ppo_model.actor(obs_t)
            action = torch.tanh(mean) * ppo_model.act_scale + ppo_model.act_bias
        return action.squeeze(0).cpu().numpy()

    T_air, time, T_out, Q_SG, action_list, energy_list, penalty_list, \
    temp_metric_list, lb_list, ub_list = ddpg_online(
        env=env,
        env_idx=0,
        policy_file=None,          # 不再加载老的 TF meta ckpt
        init_policy=init_policy,   # 改成 PPO warm start
        start=online_start,
        end=online_end,
        gamma=1.0,
        epochs=10,
        pi_lr=1e-3,
        q_lr=1e-3,
        hidden_sizes=(64, 64, 64, 64),
        activation=tf.nn.relu,
        max_ep_len=4320,
        save_freq=5,
        steps_per_epoch=4320,
        replay_size=int(1e8),
        polyak=0.995,
        batch_size=100,
        update_after=6,
        update_every=12,
        act_noise=0.001,
        rand_act_ratio=0.,
        warmstart_steps=1000
    )

    print("Energy Use in kWh: %.2f" % float(energy_list[-1]))
    print("# of Hours out of Bounds: %.2f" % float(penalty_list[-1]))
    print("Temperature Exceedance in degC-hr: %.2f" % float(temp_metric_list[-1]))

    d = {
        'energy_meta': energy_list,
        'penalty_meta': penalty_list,
        'exceedance_meta': temp_metric_list
    }
    df = pd.DataFrame(data=d)
    df.to_csv('results_meta.csv', index=False)


if __name__ == "__main__":
    main()
