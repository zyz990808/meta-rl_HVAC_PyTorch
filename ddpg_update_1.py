import torch
import numpy as np
import os

from Env_develop import ContinuousBuildingControlEnvironment as BEnv
from ddpg_torch_1 import ddpg_torch
from train_ppo_cps import ActorCritic


def main():

    # =========================
    # 1️⃣ Create environment
    # =========================
    data_file = "weather_data_2013_to_2017_summer_pandas.csv"

    env = BEnv(
        data_file=data_file,
        dt=1800.0,
        start=17664,
        end=19872.5,
        C_env=3.1996e6,
        C_air=3.5187e5,
        R_rc=0.00706,
        R_oe=0.02707,
        R_er=0.00369
    )

    print(">>> Environment ready")

    # =========================
    # 2️⃣ Load meta-policy
    # =========================
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    act_low = env.action_space.low.astype(np.float32)
    act_high = env.action_space.high.astype(np.float32)

    model = ActorCritic(obs_dim, act_dim, act_low, act_high)

    model_path = "/Users/zhangyizhong/Desktop/meta_rl-develop/model/best_model.pt"

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])

    model.eval()

    print(">>> Loaded META policy")

    # =========================
    # 3️⃣ DDPG adaptation
    # =========================
    result = ddpg_torch(
        env,
        ppo_actor=model.actor,
        steps=200000
    )

    best_actor = result["best_actor"]
    final_actor = result["final_actor"]
    best_reward = result["best_reward"]
    returns = result["episode_returns"]

    print("\n>>> Best Adaptation Reward:", best_reward)
    print(">>> Final Episode Reward:", returns[-1] if len(returns) > 0 else None)

    # =========================
    # 4️⃣ Save models
    # =========================
    save_dir = "/Users/zhangyizhong/Desktop/meta_rl-develop/model"
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, "ddpg_adapted_1.pt")

    torch.save({
        "best_actor": best_actor.state_dict(),
        "final_actor": final_actor.state_dict(),
        "best_reward": best_reward,
        "returns": returns
    }, save_path)

    print(f">>> Saved models to {save_path}")


if __name__ == "__main__":
    main()