# =========================
# MULTI-ENV PPO + DDPG (SOFT DDPG UPDATE VERSION)
# =========================

import os
import random
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
BASE_DIR = "/Users/zhangyizhong/Desktop/meta_rl-develop_9"
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

DDPG_EVAL_STEPS = 100000

DDPG_ACCEPT_DELTA = 0.2
DDPG_SOFT_UPDATE_ALPHA = 0.05


# =========================
# DDPG → PPO soft update
# =========================
def soft_update_ddpg_to_ppo(ddpg_actor, ppo_actor, alpha=0.1):
    ddpg_layers = [m for m in ddpg_actor.modules() if isinstance(m, torch.nn.Linear)]
    ppo_layers = [m for m in ppo_actor.modules() if isinstance(m, torch.nn.Linear)]

    for src, dst in zip(ddpg_layers, ppo_layers):
        dst.weight.data.copy_(
            (1.0 - alpha) * dst.weight.data + alpha * src.weight.data
        )
        dst.bias.data.copy_(
            (1.0 - alpha) * dst.bias.data + alpha * src.bias.data
        )


# =========================
# PPO evaluation
# =========================
def eval_ppo(env, model, device, steps=720):
    obs = env.reset()
    total_reward = 0.0

    for _ in range(steps):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            action, _, _ = model.get_action_and_value(obs_t)
            action = action.squeeze(0).cpu().numpy()

        obs, reward, done, _ = env.step(action)
        total_reward += reward

        if done:
            break

    return total_reward


# =========================
# ENV CREATION
# =========================
def create_env_list_from_csv(EnvClass, csv_path, n=1):
    df = pd.read_csv(csv_path)
    sampled_df = df.sample(n=n, random_state=SEED)

    envs = []
    for local_id, (i, row) in enumerate(sampled_df.iterrows()):
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
        env.seed(SEED + local_id)
        envs.append(env)

    print(f"Created {len(envs)} environments")
    return envs


# =========================
# TRAINING
# =========================
def train_multi_env(env_list, model, optimizer, device):
    os.makedirs(SAVE_DIR, exist_ok=True)

    buffer = RolloutBuffer(
        int(np.prod(env_list[0].observation_space.shape)),
        int(np.prod(env_list[0].action_space.shape)),
        ROLLOUT_STEPS,
        device,
    )

    num_updates = TOTAL_TIMESTEPS // ROLLOUT_STEPS

    update_list = []
    ppo_return_list = []
    final_return_list = []

    best_return = -1e9

    for update in range(1, num_updates + 1):

        env = random.choice(env_list)
        obs = env.reset()
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device)

        buffer.reset()

        # ===== PPO rollout =====
        for _ in range(ROLLOUT_STEPS):

            with torch.no_grad():
                action_t, logprob_t, value_t = model.get_action_and_value(
                    obs_t.unsqueeze(0)
                )
                action_t = action_t.squeeze(0)
                logprob_t = logprob_t.squeeze(0)
                value_t = value_t.squeeze(0)

            action = action_t.cpu().numpy()
            next_obs, reward, done, _ = env.step(action)

            buffer.add(
                obs_t,
                action_t,
                logprob_t,
                torch.tensor(reward, device=device),
                torch.tensor(float(done), device=device),
                value_t,
                torch.tensor(0.0, device=device),
                torch.tensor(0.0, device=device),
            )

            obs = next_obs
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)

            if done:
                obs = env.reset()
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device)

        # ===== PPO update =====
        with torch.no_grad():
            last_value = model.get_value(obs_t.unsqueeze(0)).squeeze(0)

        buffer.compute_returns_and_advantages(last_value, GAMMA, GAE_LAMBDA)

        adv = buffer.advantages
        buffer.advantages = (adv - adv.mean()) / (adv.std() + 1e-8)

        for _ in range(NUM_EPOCHS):
            for mb in buffer.get_minibatches(MINIBATCH_SIZE):

                new_logprob, entropy, new_value = model.evaluate_actions(
                    mb[0], mb[1]
                )
                ratio = (new_logprob - mb[2]).exp()

                pg_loss = torch.max(
                    -mb[3] * ratio,
                    -mb[3] * torch.clamp(
                        ratio,
                        1 - CLIP_COEF,
                        1 + CLIP_COEF,
                    ),
                ).mean()

                v_loss = 0.5 * (mb[4] - new_value).pow(2).mean()
                loss = pg_loss + VF_COEF * v_loss - ENT_COEF * entropy.mean()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()

        # ===== DDPG adaptation =====
        adapted_actor, adapted_return = ddpg_adapt(
            env,
            ppo_actor=model.actor,
            device=device,
            steps=DDPG_EVAL_STEPS,
        )

        ppo_eval_return = eval_ppo(env, model, device)

        # ===== Conservative selection =====
        if adapted_return > ppo_eval_return + DDPG_ACCEPT_DELTA:
            print(
                f">> Accept DDPG policy with soft update "
                f"alpha={DDPG_SOFT_UPDATE_ALPHA}"
            )

            soft_update_ddpg_to_ppo(
                adapted_actor,
                model.actor,
                alpha=DDPG_SOFT_UPDATE_ALPHA,
            )

            final_return = adapted_return

        else:
            print(">> Keep PPO policy")
            final_return = ppo_eval_return

        # ===== Save BEST =====
        if final_return > best_return:
            best_return = final_return
            torch.save(
                model.actor.state_dict(),
                os.path.join(SAVE_DIR, "best_actor.pth"),
            )
            print(">> Saved BEST model")

        update_list.append(update)
        ppo_return_list.append(ppo_eval_return)
        final_return_list.append(final_return)

        print(
            f"[Update {update:04d}] "
            f"PPO={ppo_eval_return:.2f} "
            f"DDPG={adapted_return:.2f} "
            f"FINAL={final_return:.2f}"
        )

    # ===== Save FINAL =====
    torch.save(
        model.actor.state_dict(),
        os.path.join(SAVE_DIR, "final_actor.pth"),
    )
    print(">> Saved FINAL model")

    # ===== Plot =====
    plt.figure()
    plt.plot(update_list, ppo_return_list, label="PPO")
    plt.plot(update_list, final_return_list, label="Final Policy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(SAVE_DIR, "final_curve.png"))
    plt.show()


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    set_seed(SEED)

    env_module = __import__(ENV_PY_NAME)
    EnvClass = getattr(env_module, "ContinuousBuildingControlEnvironment")

    env_list = create_env_list_from_csv(EnvClass, CSV_FILE, n=100)

    device = torch.device(DEVICE)

    obs_dim = int(np.prod(env_list[0].observation_space.shape))
    act_dim = int(np.prod(env_list[0].action_space.shape))

    act_low = env_list[0].action_space.low.astype(np.float32)
    act_high = env_list[0].action_space.high.astype(np.float32)

    model = ActorCritic(obs_dim, act_dim, act_low, act_high).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, eps=1e-5)

    train_multi_env(env_list, model, optimizer, str(device))