import torch
import numpy as np
import os

from Env_develop import ContinuousBuildingControlEnvironment as BEnv
from ddpg_torch import ddpg_torch
from train_ppo_cps import ActorCritic


def main():

    # =========================
    # 1. Create new environment
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

    print(">>> New environment ready")

    # =========================
    # 2. Load general policy
    # =========================
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    act_low = env.action_space.low.astype(np.float32)
    act_high = env.action_space.high.astype(np.float32)

    model = ActorCritic(obs_dim, act_dim, act_low, act_high)

    model_path = "/Users/zhangyizhong/Desktop/meta_rl-develop_9/model/final_actor.pth"

    actor_state = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False
    )

    model.actor.load_state_dict(actor_state)
    model.eval()

    print(">>> Loaded general PPO actor")

    # =========================
    # 3. DDPG adaptation setting
    # =========================
    total_steps = 100000
    warmup_steps = 80000

    print(">>> DDPG adaptation starts")
    print(">>> Using random Q")
    print(f">>> Warm-up steps: {warmup_steps}")

    # =========================
    # 4. DDPG adaptation
    # =========================
    result = ddpg_torch(
        env,
        ppo_actor=model.actor,
        steps=total_steps,
        warmup_steps=warmup_steps
    )

    best_actor = result["best_actor"]
    final_actor = result["final_actor"]
    best_reward = result["best_reward"]
    returns = result["episode_returns"]

    last_best_actor = result["last_best_actor"]
    last_min_exceed_actor = result["last_min_exceed_actor"]

    last_best_reward = result["last_best_reward"]
    last_min_exceedance = result["last_min_exceedance"]

    # =========================
    # 5. Print results
    # =========================
    print("\n=========================")
    print(">>> Best Adaptation Reward:", best_reward)

    if len(returns) > 0:
        print(">>> Final Episode Reward:", returns[-1])
    else:
        print(">>> No episode finished")

    print(">>> Last Best Reward:", last_best_reward)
    print(">>> Last Min Exceedance:", last_min_exceedance)
    print("=========================\n")

    # =========================
    # 6. Save adapted models
    # =========================
    save_dir = "/Users/zhangyizhong/Desktop/meta_rl-develop_9/ddpg"
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, "ddpg_adapted.pt")

    torch.save({
        "best_actor": best_actor.state_dict(),
        "final_actor": final_actor.state_dict(),

        "last_best_actor": last_best_actor.state_dict(),
        "last_min_exceed_actor": last_min_exceed_actor.state_dict(),

        "best_reward": best_reward,
        "last_best_reward": last_best_reward,
        "last_min_exceedance": last_min_exceedance,

        "returns": returns,
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "critic_init": "random"
    }, save_path)

    print(f">>> Saved adapted models to {save_path}")

    # =========================
    # 7. Print full return curve
    # =========================
    print("\n>>> Episode Returns:")
    for i, r in enumerate(returns):
        print(f"Episode {i+1}: {r:.2f}")


if __name__ == "__main__":
    main()