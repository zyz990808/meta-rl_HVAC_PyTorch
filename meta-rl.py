# =========================
# MULTI-ENV PPO + DDPG EVALUATION
# =========================

import os
import random
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from train_ppo_cps import ActorCritic, RolloutBuffer, set_seed
from ddpg_meta import ddpg_adapt


# =========================
# SETTINGS
# =========================
BASE_DIR = "/Users/zhangyizhong/Desktop/meta_rl-develop"
SAVE_DIR = os.path.join(BASE_DIR, "model")

ENV_PY_NAME = "Env_develop"
DATA_FILE = "weather_data_2013_to_2017_summer_pandas.csv"
CSV_FILE = "env_param.csv"

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TOTAL_TIMESTEPS = 300_000
ROLLOUT_STEPS = 2048
NUM_EPOCHS = 10
MINIBATCH_SIZE = 256

GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_COEF = 0.2
VF_COEF = 0.5
ENT_COEF = 0.01
MAX_GRAD_NORM = 0.5
LR = 3e-4
ANNEAL_LR = True
TARGET_KL = 0.02

# DDPG evaluation settings
DDPG_EVAL_STEPS = 1000


# =========================
# ENV CREATION
# =========================
def create_env_list_from_csv(EnvClass, csv_path, n=1):
    df = pd.read_csv(csv_path)
    sampled_df = df.sample(n=n, random_state=SEED)

    envs = []

    for i, row in sampled_df.iterrows():
        env = EnvClass(
            data_file=DATA_FILE,
            dt=1800.0,
            start=0.0,
            end=720.0,
            C_env=row["C_env"],
            C_air=row["C_air"],
            R_rc=row["R_rc"],
            R_oe=row["R_oe"],
            R_er=row["R_er"],
        )

        env.seed(SEED + i)
        envs.append(env)

    print(f"Created {len(envs)} environments")
    return envs


# =========================
# TRAINING
# =========================
def train_multi_env(env_list, model, optimizer, device):
    os.makedirs(SAVE_DIR, exist_ok=True)

    obs_dim = int(np.prod(env_list[0].observation_space.shape))
    act_dim = int(np.prod(env_list[0].action_space.shape))

    buffer = RolloutBuffer(obs_dim, act_dim, ROLLOUT_STEPS, device)

    global_step = 0
    episode_rows: List[Dict[str, Any]] = []
    episode_idx = 0

    num_updates = TOTAL_TIMESTEPS // ROLLOUT_STEPS

    best_adapted_return = -1e9

    update_list = []
    avg_return_list = []
    adapted_return_list = []

    for update in range(1, num_updates + 1):

        # use one environment per update
        env = random.choice(env_list)

        obs = env.reset()
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device)

        ep_return = 0.0
        ep_len = 0
        ep_energy = 0.0
        ep_comfort = 0.0

        if ANNEAL_LR:
            frac = 1.0 - (update - 1.0) / num_updates
            for pg in optimizer.param_groups:
                pg["lr"] = frac * LR

        buffer.reset()

        # ===== PPO rollout with PPO model itself =====
        for _ in range(ROLLOUT_STEPS):
            global_step += 1

            with torch.no_grad():
                action_t, logprob_t, value_t = model.get_action_and_value(obs_t.unsqueeze(0))
                action_t = action_t.squeeze(0)
                logprob_t = logprob_t.squeeze(0)
                value_t = value_t.squeeze(0)

            action = action_t.cpu().numpy()
            next_obs, reward, done, info = env.step(action)

            en = float(info.get("EnergyNorm", 0.0))
            co = float(info.get("TempNorm", 0.0))

            ep_return += reward
            ep_len += 1
            ep_energy += en
            ep_comfort += co

            buffer.add(
                obs_t,
                action_t,
                logprob_t,
                torch.tensor(reward, device=device),
                torch.tensor(float(done), device=device),
                value_t,
                torch.tensor(en, device=device),
                torch.tensor(co, device=device),
            )

            obs = next_obs
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)

            if done:
                episode_rows.append({
                    "episode": episode_idx,
                    "return": ep_return,
                    "energy": ep_energy,
                    "comfort": ep_comfort,
                    "length": ep_len
                })

                episode_idx += 1
                ep_return = 0.0
                ep_len = 0
                ep_energy = 0.0
                ep_comfort = 0.0

                obs = env.reset()
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device)

        # ===== PPO update =====
        with torch.no_grad():
            last_value = model.get_value(obs_t.unsqueeze(0)).squeeze(0)

        buffer.compute_returns_and_advantages(last_value, GAMMA, GAE_LAMBDA)

        adv = buffer.advantages
        buffer.advantages = (adv - adv.mean()) / (adv.std() + 1e-8)

        approx_kl = 0.0
        pi_losses = []
        v_losses = []
        entropies = []

        for _ in range(NUM_EPOCHS):
            for (
                mb_obs,
                mb_actions,
                mb_logprob_old,
                mb_adv,
                mb_returns,
                mb_values_old
            ) in buffer.get_minibatches(MINIBATCH_SIZE):

                new_logprob, entropy, new_value = model.evaluate_actions(mb_obs, mb_actions)

                ratio = (new_logprob - mb_logprob_old).exp()

                pg_loss = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * torch.clamp(ratio, 1 - CLIP_COEF, 1 + CLIP_COEF)
                ).mean()

                v_loss = 0.5 * (mb_returns - new_value).pow(2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss + VF_COEF * v_loss - ENT_COEF * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()

                approx_kl = (mb_logprob_old - new_logprob).mean().item()

                pi_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                entropies.append(entropy_loss.item())

            if TARGET_KL and approx_kl > TARGET_KL:
                break

        # ===== DDPG evaluation on the same env =====
        _, adapted_episode_returns, _ = ddpg_adapt(
            env,
            ppo_actor=model.actor,
            device=device,
            steps=DDPG_EVAL_STEPS
        )

        if len(adapted_episode_returns) > 0:
            adapted_return = float(np.mean(adapted_episode_returns))
        else:
            adapted_return = float("-inf")

        # ===== logging =====
        avg10 = np.mean([r["return"] for r in episode_rows[-10:]]) if episode_rows else 0
        avg_energy = np.mean([r["energy"] for r in episode_rows[-10:]]) if episode_rows else 0
        avg_comfort = np.mean([r["comfort"] for r in episode_rows[-10:]]) if episode_rows else 0

        update_list.append(update)
        avg_return_list.append(avg10)
        adapted_return_list.append(adapted_return)

        # ===== save best model by adapted return =====
        if update == 1 or adapted_return > best_adapted_return:
            best_adapted_return = adapted_return
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "update": update,
                "avg_return": avg10,
                "adapted_return": adapted_return
            }, os.path.join(SAVE_DIR, "best_model.pt"))

        print(
            f"[Update {update:04d}/{num_updates}] "
            f"steps={global_step} episodes={episode_idx} "
            f"avgReturn={avg10:.3f} | AdaptedReturn={adapted_return:.3f} "
            f"| EnergyNorm={avg_energy:.3f} TempNorm={avg_comfort:.3f} "
            f"| piLoss={np.mean(pi_losses):.4f} vLoss={np.mean(v_losses):.4f} "
            f"entropy={np.mean(entropies):.4f} KL={approx_kl:.4f}"
        )

    # ===== save final model =====
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "update": update,
        "avg_return": avg10,
        "adapted_return": adapted_return
    }, os.path.join(SAVE_DIR, "final_model.pt"))

    # ===== plot PPO and adapted returns =====
    plt.figure()
    plt.plot(update_list, avg_return_list, label="PPO avgReturn")
    plt.plot(update_list, adapted_return_list, label="After Adaptation")
    plt.xlabel("Update")
    plt.ylabel("Return")
    plt.title("Training Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(SAVE_DIR, "compare_curve.png"))
    plt.show()


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    set_seed(SEED)

    env_module = __import__(ENV_PY_NAME)
    EnvClass = getattr(env_module, "ContinuousBuildingControlEnvironment")

    env_list = create_env_list_from_csv(EnvClass, CSV_FILE, n=5)

    device = torch.device(DEVICE)

    obs_dim = int(np.prod(env_list[0].observation_space.shape))
    act_dim = int(np.prod(env_list[0].action_space.shape))

    act_low = env_list[0].action_space.low.astype(np.float32)
    act_high = env_list[0].action_space.high.astype(np.float32)

    model = ActorCritic(obs_dim, act_dim, act_low, act_high).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, eps=1e-5)

    train_multi_env(env_list, model, optimizer, str(device))