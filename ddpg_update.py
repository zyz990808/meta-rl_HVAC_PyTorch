import torch
import numpy as np

# ===== NEW ENV =====
from Env_develop import ContinuousBuildingControlEnvironment as BEnv

# ===== DDPG =====
from ddpg_torch import ddpg_torch

# ===== PPO MODEL =====
from train_ppo_cps import ActorCritic


def main():

    # =========================
    # 1️⃣ Create new environment (for generalization test)
    # =========================
    data_file = "weather_data_2013_to_2017_summer_pandas.csv"

    env = BEnv(
        data_file=data_file,
        dt=1800.0,
        start=6000,
        end=8160,

        # 👉 A fixed test environment (can be changed)
        C_env=3.1996e6,
        C_air=3.5187e5,
        R_rc=0.00706,
        R_oe=0.02707,
        R_er=0.00369
    )

    print(">>> Environment ready")

    # =========================
    # 2️⃣ Load meta-policy (your final model)
    # =========================
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    act_low = env.action_space.low.astype(np.float32)
    act_high = env.action_space.high.astype(np.float32)

    model = ActorCritic(obs_dim, act_dim, act_low, act_high)

    model_path = "/Users/zhangyizhong/Desktop/meta_rl-develop/model/best_model.pt"
    model.load_state_dict(torch.load(model_path, map_location="cpu"))

    model.eval()

    print(">>> Loaded META policy from final_model.pt")

    # =========================
    # 3️⃣ DDPG adaptation
    # =========================
    reward = ddpg_torch(
        env,
        ppo_actor=model.actor,   # 👈 Initialize with PPO actor
        steps=30000
    )

    print("\n>>> Final Adaptation Reward:", reward)


if __name__ == "__main__":
    main()


