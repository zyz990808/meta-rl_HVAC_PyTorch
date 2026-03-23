import torch
import numpy as np
import matplotlib.pyplot as plt

from env_old import ContinuousBuildingControlEnvironment as BEnv
from ddpg_torch import Actor


def test_policy_torch(policy_path):

    data_file = 'weather_data_2013_to_2017_summer_pandas.csv'

    start_time = 17664.
    end_time = 19872.5

    env = BEnv(
        data_file,
        start=start_time,
        end=end_time,
        C_env=3.1996e6,
        C_air=3.5187e5,
        R_rc=0.00706,
        R_oe=0.02707,
        R_er=0.00369,
        lb_set=22.,
        ub_set=24.
    )

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_low = env.action_space.low
    act_high = env.action_space.high

    # ===== load model =====
    model = Actor(obs_dim, act_dim, act_low, act_high)
    model.load_state_dict(torch.load(policy_path, map_location="cpu"))
    model.eval()

    print(">>> Loaded policy")

    # ===== containers (full range) =====
    obs_list = []
    reward_list = []
    energy_list = [0]
    penalty_list = [0]
    temp_metric_list = [0]
    action_list = []
    lb_list = []
    ub_list = []

    obs = env.reset()
    done = False

    step_count = 0

    # ===== rollout =====
    while not done:

        step_count += 1
        obs_list.append(obs.copy())

        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            action = model(obs_t).squeeze(0).numpy()

        obs, reward, done, dic = env.step(action)

        reward_list.append(reward)

        action_list.append(dic['a_t'] if isinstance(dic['a_t'], float) else dic['a_t'][0])

        # ===== energy =====
        energy = dic['Energy'] if isinstance(dic['Energy'], float) else dic['Energy'][0]
        energy_list.append(energy + energy_list[-1])

        penalty_list.append(dic['Penalty'] + penalty_list[-1])
        temp_metric_list.append(dic['Exceedance'] + temp_metric_list[-1])

        lb_list.append(dic['lb'])
        ub_list.append(dic['ub'])

    env.close()

    print("Total rollout steps:", step_count)

    # ===== denormalization =====
    low = np.array([10.0, 18.0, 21.0, -40.0, 0., 50., 0])
    high = np.array([35.0, 27.0, 23.0, 40.0, 1100., 180., 23])

    obs_arr = np.array(obs_list)

    T_air_full = obs_arr[:, 1] * (high[1] - low[1]) + low[1]
    T_out_full = obs_arr[:, 3] * (high[3] - low[3]) + low[3]
    Q_SG_full = obs_arr[:, 4] * (high[4] - low[4]) + low[4]

    time_full = np.linspace(start_time, end_time, len(T_air_full))

    # =====================================================
    # 🔥 fixed plot range: 18700 → 18820
    # =====================================================
    plot_start_time = 18700
    plot_end_time = 18820

    mask = (time_full >= plot_start_time) & (time_full <= plot_end_time)

    T_air = T_air_full[mask]
    T_out = T_out_full[mask]
    Q_SG = Q_SG_full[mask]
    action_plot = np.array(action_list)[mask]
    lb_plot = np.array(lb_list)[mask]
    ub_plot = np.array(ub_list)[mask]
    time = time_full[mask]

    # ===== plotting =====
    plt.figure(figsize=(12,8))

    plt.subplot(2,2,1)
    plt.plot(time, T_air)
    plt.plot(time, lb_plot, '--')
    plt.plot(time, ub_plot, '--')
    plt.title("Indoor Air Temperature")

    plt.subplot(2,2,2)
    plt.plot(time, action_plot)
    plt.title("Thermal Input")

    plt.subplot(2,2,3)
    plt.plot(time, T_out)
    plt.title("Outdoor Temperature")

    plt.subplot(2,2,4)
    plt.plot(time, Q_SG)
    plt.title("Solar Heat Gain")

    plt.tight_layout()

    # save figure to root directory
    plt.savefig("ddpg_result.png")
    plt.close()

    print(">>> Plot saved to ddpg_result.png")

    # =====================================================
    # 🔥 full range metrics (final version)
    # =====================================================
    print("\n===== FULL METRICS =====")
    print("Energy (kWh):", int(energy_list[-1]))
    print("Out-of-bound Hours:", int(penalty_list[-1]))
    print("Temperature Exceedance:", int(temp_metric_list[-1]))
    print("Avg reward:", np.mean(reward_list))
    print("Total steps:", len(reward_list))


# ===== main =====
if __name__ == "__main__":
    test_policy_torch("model/multi_env_meta/ddpg_actor_final.pth")