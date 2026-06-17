import numpy as np
import os
import pandas as pd
import torch
import matplotlib.pyplot as plt

from Env_develop import ContinuousBuildingControlEnvironment as BEnv
from ddpg_torch import Actor


# =========================
# CUSTOM PLOT
# =========================
def custom_plot(
    T_air, time, T_out, Q_SG,
    sat_list, zat_list, energy_list, penalty_list,
    lb_list, ub_list, idx, folder_name
):

    save_dir = os.path.join("plots", folder_name)
    os.makedirs(save_dir, exist_ok=True)

    fig, axs = plt.subplots(2, 2, figsize=(10, 7))

    axs[0, 0].plot(time, T_air)
    axs[0, 0].plot(time, lb_list, "--")
    axs[0, 0].plot(time, ub_list, "--")
    axs[0, 0].set_title("Indoor Air Temperature")

    axs[0, 1].plot(time, sat_list, label="SAT")
    axs[0, 1].plot(time, zat_list, label="ZAT")
    axs[0, 1].set_title("Thermal Input")
    axs[0, 1].legend()

    axs[1, 0].plot(time, T_out)
    axs[1, 0].set_title("Outdoor Temperature")

    axs[1, 1].plot(time, Q_SG)
    axs[1, 1].set_title("Solar Heat Gain")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{idx}.png"))
    plt.show()


# =========================
# TEST POLICY
# =========================
def test_policy(
    policy_file,
    actor_name="best_actor",
    start=17664.,
    end=19872.5,
    data_file="weather_data_2013_to_2017_summer_pandas.csv"
):

    env = BEnv(
        data_file=data_file,
        start=start,
        end=end,
        C_env=3.1996e6,
        C_air=3.5187e5,
        R_rc=0.00706,
        R_oe=0.02707,
        R_er=0.00369,
        lb_set=22.,
        ub_set=24.,
    )

    obs, done = env.reset(), False

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    act_low = env.action_space.low.astype(np.float32)
    act_high = env.action_space.high.astype(np.float32)

    actor = Actor(
        obs_dim,
        act_dim,
        act_low,
        act_high,
    )

    checkpoint = torch.load(
        policy_file,
        map_location="cpu",
        weights_only=False,
    )

    if actor_name not in checkpoint:
        raise KeyError(
            f"{actor_name} not found in checkpoint. "
            f"Available keys: {list(checkpoint.keys())}"
        )

    actor.load_state_dict(checkpoint[actor_name])
    actor.eval()

    print(f">>> Loaded actor: {actor_name}")
    print(f">>> Action low: {act_low}")
    print(f">>> Action high: {act_high}")

    obs_list = []
    reward_list = []

    sat_list = []
    zat_list = []

    raw_action_list = []

    energy_list = [0]
    penalty_list = [0]
    temp_metric_list = [0]

    lb_list = []
    ub_list = []

    with torch.no_grad():
        while True:

            if not done:
                obs_list.append(obs.copy())

                obs_t = torch.tensor(
                    obs,
                    dtype=torch.float32,
                ).reshape(1, -1)

                action_val = actor(obs_t)
                action_val = action_val.cpu().numpy()[0]

                raw_action_list.append(action_val.copy())

                obs, reward, done, dic = env.step(action_val)

                reward_list.append(reward)

                sat_list.append(dic["SAT_sp"])
                zat_list.append(dic["ZAT_sp_used"])

                energy_list.append(
                    dic["TotalEnergy_kWh"] + energy_list[-1]
                )

                penalty_step = 0.5 if dic["TempExceed_degC"] > 0 else 0
                penalty_list.append(
                    penalty_step + penalty_list[-1]
                )

                temp_metric_list.append(
                    dic["TempExceed_degC"] * 0.5
                    + temp_metric_list[-1]
                )

                lb_list.append(env.lb)
                ub_list.append(env.ub)

            if done:
                break

    env.close()

    low = env.low
    high = env.high

    obs_arr = np.array(obs_list)

    T_air = obs_arr[:, 1] * (high[1] - low[1]) + low[1]
    time = np.linspace(start, end, len(T_air))
    T_out = obs_arr[:, 3] * (high[3] - low[3]) + low[3]
    Q_SG = obs_arr[:, 4] * (high[4] - low[4]) + low[4]

    raw_action_arr = np.array(raw_action_list)

    print(">>> Raw action min:", raw_action_arr.min(axis=0))
    print(">>> Raw action max:", raw_action_arr.max(axis=0))
    print(">>> Raw action mean:", raw_action_arr.mean(axis=0))

    return (
        T_air, time, T_out, Q_SG,
        np.array(sat_list), np.array(zat_list),
        np.array(energy_list[1:]),
        np.array(penalty_list[1:]),
        np.array(temp_metric_list[1:]),
        lb_list, ub_list
    )


# =========================
# MAIN
# =========================
def main():

    torch.manual_seed(0)
    np.random.seed(0)

    data_file = "weather_data_2013_to_2017_summer_pandas.csv"

    policy_file = "/Users/zhangyizhong/Desktop/meta_rl-develop_10/ddpg/ddpg_adapted.pt"

    actor_name = "best_actor"
    # actor_name = "final_actor"
    # actor_name = "last_best_actor"
    # actor_name = "last_min_exceed_actor"

    for idx in range(0, 1):

        (
            T_air, time, T_out, Q_SG,
            sat_list, zat_list,
            energy_list, penalty_list,
            temp_metric_list,
            lb_list, ub_list
        ) = test_policy(
            start=17664.,
            end=19872.5,
            data_file=data_file,
            policy_file=policy_file,
            actor_name=actor_name,
        )

        custom_plot(
            T_air[2100:2300],
            time[2100:2300],
            T_out[2100:2300],
            Q_SG[2100:2300],
            sat_list[2100:2300],
            zat_list[2100:2300],
            energy_list[2100:2300],
            penalty_list[2100:2300],
            lb_list[2100:2300],
            ub_list[2100:2300],
            idx,
            "offline",
        )

        print(idx)

    d = {
        "energy_true": energy_list,
        "penalty_true": penalty_list,
        "exceedance_true": temp_metric_list,
    }

    pd.DataFrame(d).to_csv(
        "results_true.csv",
        index=False,
    )

    d_profile = {
        "lb": lb_list,
        "ub": ub_list,
        "Qsg": Q_SG,
        "Tout": T_out,
        "Tair_true": T_air,
        "sat_true": sat_list,
        "zat_true": zat_list,
    }

    pd.DataFrame(d_profile).to_csv(
        "results_profile.csv",
        index=False,
    )

    print("Tested actor:", actor_name)
    print("Energy Use in kWh: %.2f" % energy_list[-1])
    print("# of Hours out of Bounds: %.2f" % penalty_list[-1])
    print("Temperature Exceedance in degC-hr: %.2f" % temp_metric_list[-1])


if __name__ == "__main__":
    main()



