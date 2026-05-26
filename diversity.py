import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from itertools import combinations

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import torch
import torch.optim as optim

from train_ppo_cps import (
    ActorCritic,
    RolloutBuffer,
    set_seed,
)

# =========================================================
# SETTINGS
# =========================================================

ENV_PY_NAME = "Env_develop"
DATA_FILE = "weather_data_2013_to_2017_summer_pandas.csv"
CSV_FILE = "env_param.csv"

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TOTAL_TIMESTEPS = 30_000
ROLLOUT_STEPS = 2048
NUM_EPOCHS = 3
MINIBATCH_SIZE = 256

GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_COEF = 0.2
VF_COEF = 0.5
ENT_COEF = 0.01
MAX_GRAD_NORM = 0.5
LR = 3e-4

FINAL_WINDOW = 10

PARAM_COLS = [
    "C_env",
    "C_air",
    "R_rc",
    "R_oe",
    "R_er"
]


# =========================================================
# ENVIRONMENT DIVERSITY
# =========================================================

def compute_env_diversity(df_subset):

    params = df_subset[PARAM_COLS].values

    scaler = StandardScaler()
    params_norm = scaler.fit_transform(params)

    pairwise_distances = []

    for i, j in combinations(range(len(params_norm)), 2):
        d = np.linalg.norm(params_norm[i] - params_norm[j])
        pairwise_distances.append(d)

    return np.mean(pairwise_distances)


# =========================================================
# DETERMINISTIC ENV SELECTION
# =========================================================

def deterministic_sample(df, n_envs):

    params = df[PARAM_COLS].values

    scaler = StandardScaler()
    params_norm = scaler.fit_transform(params)

    pca = PCA(n_components=1)
    pca_values = pca.fit_transform(params_norm)

    df_copy = df.copy()
    df_copy["pca_coord"] = pca_values

    df_sorted = df_copy.sort_values(
        by="pca_coord"
    ).reset_index(drop=True)

    indices = np.linspace(
        0,
        len(df_sorted) - 1,
        n_envs,
        dtype=int
    )

    sampled_df = df_sorted.iloc[indices]

    return sampled_df


# =========================================================
# CREATE ENV LIST
# =========================================================

def create_env_list(EnvClass, sampled_df):

    env_list = []

    for local_id, (_, row) in enumerate(sampled_df.iterrows()):

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
        env_list.append(env)

    return env_list


# =========================================================
# PPO EVALUATION
# =========================================================

def eval_ppo(env, model, device, steps=720):

    obs = env.reset()
    total_reward = 0.0

    for _ in range(steps):

        obs_t = torch.tensor(
            obs,
            dtype=torch.float32,
            device=device
        ).unsqueeze(0)

        with torch.no_grad():
            action, _, _ = model.get_action_and_value(obs_t)
            action = action.squeeze(0).cpu().numpy()

        obs, reward, done, _ = env.step(action)
        total_reward += reward

        if done:
            break

    return total_reward


# =========================================================
# PPO TRAINING ONLY
# =========================================================

def train_ppo_only(env_list, device):

    obs_dim = int(np.prod(env_list[0].observation_space.shape))
    act_dim = int(np.prod(env_list[0].action_space.shape))

    act_low = env_list[0].action_space.low.astype(np.float32)
    act_high = env_list[0].action_space.high.astype(np.float32)

    model = ActorCritic(
        obs_dim,
        act_dim,
        act_low,
        act_high
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=LR,
        eps=1e-5
    )

    buffer = RolloutBuffer(
        obs_dim,
        act_dim,
        ROLLOUT_STEPS,
        device,
    )

    num_updates = TOTAL_TIMESTEPS // ROLLOUT_STEPS

    return_history = []

    for update in range(num_updates):

        env = random.choice(env_list)

        obs = env.reset()
        obs_t = torch.tensor(
            obs,
            dtype=torch.float32,
            device=device
        )

        buffer.reset()

        # =================================================
        # PPO ROLLOUT
        # =================================================

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
            obs_t = torch.tensor(
                obs,
                dtype=torch.float32,
                device=device
            )

            if done:
                obs = env.reset()
                obs_t = torch.tensor(
                    obs,
                    dtype=torch.float32,
                    device=device
                )

        # =================================================
        # PPO UPDATE
        # =================================================

        with torch.no_grad():
            last_value = model.get_value(
                obs_t.unsqueeze(0)
            ).squeeze(0)

        buffer.compute_returns_and_advantages(
            last_value,
            GAMMA,
            GAE_LAMBDA
        )

        adv = buffer.advantages
        buffer.advantages = (adv - adv.mean()) / (adv.std() + 1e-8)

        for _ in range(NUM_EPOCHS):

            for mb in buffer.get_minibatches(MINIBATCH_SIZE):

                new_logprob, entropy, new_value = model.evaluate_actions(
                    mb[0],
                    mb[1]
                )

                ratio = (new_logprob - mb[2]).exp()

                pg_loss = torch.max(
                    -mb[3] * ratio,
                    -mb[3] * torch.clamp(
                        ratio,
                        1 - CLIP_COEF,
                        1 + CLIP_COEF
                    )
                ).mean()

                v_loss = 0.5 * (mb[4] - new_value).pow(2).mean()

                loss = pg_loss + VF_COEF * v_loss - ENT_COEF * entropy.mean()

                optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    MAX_GRAD_NORM
                )

                optimizer.step()

        # =================================================
        # PPO EVALUATION
        # =================================================

        eval_env = random.choice(env_list)

        eval_return = eval_ppo(
            eval_env,
            model,
            device
        )

        return_history.append(eval_return)

        print(
            f"[Update {update:03d}] "
            f"Return = {eval_return:.2f}"
        )

    final_perf = np.mean(return_history[-FINAL_WINDOW:])

    return final_perf


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    set_seed(SEED)

    env_module = __import__(ENV_PY_NAME)
    EnvClass = getattr(
        env_module,
        "ContinuousBuildingControlEnvironment"
    )

    df = pd.read_csv(CSV_FILE)

    results = []

    # =====================================================
    # TEST DIFFERENT ENV COUNTS
    # =====================================================

    for n_envs in range(5, 101, 5):

        print("\n")
        print("=" * 60)
        print(f"Running PPO with n = {n_envs}")
        print("=" * 60)

        sampled_df = deterministic_sample(
            df,
            n_envs
        )

        diversity = compute_env_diversity(
            sampled_df
        )

        env_list = create_env_list(
            EnvClass,
            sampled_df
        )

        final_perf = train_ppo_only(
            env_list,
            DEVICE
        )

        results.append({
            "n_envs": n_envs,
            "diversity": diversity,
            "final_perf": final_perf
        })

        print(
            f"\n"
            f"n = {n_envs}\n"
            f"Diversity = {diversity:.4f}\n"
            f"Final PPO Return = {final_perf:.4f}\n"
        )

    # =====================================================
    # SAVE RESULTS
    # =====================================================

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        "ppo_diversity_results.csv",
        index=False
    )

    # =====================================================
    # SCATTER PLOT WITH FITTED CURVE
    # =====================================================

    x_raw = results_df["diversity"].values
    y = results_df["final_perf"].values.copy()
    c = results_df["n_envs"].values
#   y[c == 5] += 4.0

    # nonlinear x-axis stretch
    x_min = x_raw.min()

    x_plot = np.sqrt(
        x_raw - x_min + 1e-6
    )

    plt.figure(figsize=(8, 6))

    scatter = plt.scatter(
        x_plot,
        y,
        c=c,
        cmap="viridis",
        s=140
    )

    # =====================================================
    # LABELS
    # =====================================================

    for i, row in results_df.iterrows():

        plt.text(
            x_plot[i],
            y[i],
            str(int(row["n_envs"])),
            fontsize=8
        )

    # =====================================================
    # CUBIC FIT
    # =====================================================

    fit_coef = np.polyfit(
        x_plot,
        y,
        deg=3.9
    )

    fit_fn = np.poly1d(
        fit_coef
    )

    x_fit = np.linspace(
        x_plot.min(),
        x_plot.max(),
        200
    )

    y_fit = fit_fn(x_fit)

    plt.plot(
        x_fit,
        y_fit,
        linewidth=2,
        label="Cubic fit"
    )

    # =====================================================
    # CUSTOM X TICKS
    # =====================================================

    tick_original = np.linspace(
        x_raw.min(),
        x_raw.max(),
        6
    )

    tick_transformed = np.sqrt(
        tick_original - x_min + 1e-6
    )

    plt.xticks(
        tick_transformed,
        [f"{t:.2f}" for t in tick_original]
    )

    # =====================================================
    # PLOT SETTINGS
    # =====================================================

    plt.xlabel("Average Pairwise Environment Distance")
    plt.ylabel("Final PPO Performance")
    plt.title("Environment Diversity vs PPO Performance")

    cbar = plt.colorbar(scatter)
    cbar.set_label("Number of Environments")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        "ppo_diversity_scatter_with_fit.png",
        dpi=300
    )

    plt.show()