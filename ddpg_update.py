from env import ContinuousBuildingControlEnvironment as BEnv
import torch
from ddpg_torch import ddpg_torch
from meta_rl_policy import ActorCritic


def main():

    data_file = 'weather_data_2013_to_2017_summer_pandas.csv'

    env = BEnv(
        data_file,
        start=6000,
        end=8160,
        C_env=3.1996e6,
        C_air=3.5187e5,
        R_rc=0.00706,
        R_oe=0.02707,
        R_er=0.00369
    )

    # =========================
    # ⭐ add scaling (required)
    # =========================
    energy_scale = 1000
    exceed_scale = 10

    env.update_scale(energy_scale, exceed_scale)

    # ===== load meta-policy =====
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_low = env.action_space.low
    act_high = env.action_space.high

    ppo_model = ActorCritic(obs_dim, act_dim, act_low, act_high)
    ppo_model.load_state_dict(
        torch.load("model/multi_env_meta/ppo_multi_env_meta_final.pth")
    )
    ppo_model.eval()

    print(">>> Loaded PPO meta-policy")

    # =========================
    # ⭐ run DDPG and save best
    # =========================
    best_reward, best_actor = ddpg_torch(
        env,
        ppo_actor=ppo_model.actor,
        steps=200000,
        save_best=True
    )

    print("Best Reward:", best_reward)

    # =========================
    # ⭐ save best model
    # =========================
    torch.save(
        best_actor.state_dict(),
        "model/multi_env_meta/ddpg_actor_best.pth"
    )

    print(">>> Saved BEST DDPG model")


if __name__ == "__main__":
    main()