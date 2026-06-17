import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from itertools import combinations

from sklearn.preprocessing import StandardScaler

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

TOTAL_TIMESTEPS = 150_000
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

N_ENVS = 100
NUM_GROUPS = 10

# scale controls how widely the 100 environments spread
# scale = 0.1 means environments are close to the center
# scale = 1.0 means environments cover the full parameter range
GROUP_SCALES = np.linspace(0.1, 1.0, NUM_GROUPS)

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

def compute_env_diversity(df_subset, scaler):

    params = df_subset[PARAM_COLS].values

    # Important:
    # use global scaler fitted on the full parameter space
    params_norm = scaler.transform(params)

    pairwise_distances = []

    for i, j in combinations(range(len(params_norm)), 2):
        d = np.linalg.norm(params_norm[i] - params_norm[j])
        pairwise_distances.append(d)

    return np.mean(pairwise_distances)


# =========================================================
# STRUCTURED ENVIRONMENT GENERATION
# =========================================================

def generate_structured_env_group(df, n_envs, scale):

    param_min = df[PARAM_COLS].min()
    param_max = df[PARAM_COLS].max()

    center = (param_min + param_max) / 2.0
    half_range = (param_max - param_min) / 2.0

    # alpha controls the location of each environment
    # smaller scale -> smaller diversity
    # larger scale -> larger diversity
    alphas = np.linspace(
        -scale,
        scale,
        n_envs
    )

    rows = []

    for alpha in alphas:

        row = {}

        for col in PARAM_COLS:
            value = center[col] + alpha * half_range[col]

            # keep parameters inside the original bounds
            value = np.clip(
                value,
                param_min[col],
                param_max[col]
            )

            row[col] = value

        rows.append(row)

    sampled_df = pd.DataFrame(rows)

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

    # global scaler for all groups
    global_scaler = StandardScaler()
    global_scaler.fit(df[PARAM_COLS].values)

    results = []

    for group_id, scale in enumerate(GROUP_SCALES):

        print("\n")
        print("=" * 60)
        print(f"Running group {group_id + 1}/{NUM_GROUPS}")
        print(f"Scale = {scale:.4f}")
        print(f"Number of environments = {N_ENVS}")
        print("=" * 60)

        sampled_df = generate_structured_env_group(
            df=df,
            n_envs=N_ENVS,
            scale=scale
        )

        diversity = compute_env_diversity(
            sampled_df,
            global_scaler
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
            "group_id": group_id + 1,
            "n_envs": N_ENVS,
            "scale": scale,
            "diversity": diversity,
            "final_perf": final_perf
        })

        print(
            f"\n"
            f"Group = {group_id + 1}\n"
            f"Scale = {scale:.4f}\n"
            f"Diversity = {diversity:.4f}\n"
            f"Final PPO Return = {final_perf:.4f}\n"
        )

    # =====================================================
    # SAVE RESULTS
    # =====================================================

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        "ppo_structured_diversity_100envs_results.csv",
        index=False
    )

    # =====================================================
    # SCATTER PLOT
    # =====================================================

    x = results_df["diversity"].values
    y = results_df["final_perf"].values
    c = results_df["scale"].values

    plt.figure(figsize=(8, 6))

    scatter = plt.scatter(
        x,
        y,
        c=c,
        cmap="viridis",
        s=140
    )

    for i, row in results_df.iterrows():

        plt.text(
            row["diversity"],
            row["final_perf"],
            f"S{row['scale']:.1f}",
            fontsize=8
        )

    # optional fitted curve
    if len(x) >= 4:
        fit_coef = np.polyfit(
            x,
            y,
            deg=3
        )

        fit_fn = np.poly1d(
            fit_coef
        )

        x_fit = np.linspace(
            x.min(),
            x.max(),
            200
        )

        y_fit = fit_fn(x_fit)

        plt.plot(
            x_fit,
            y_fit,
            linewidth=2,
            label="Cubic fit"
        )

    plt.xlabel("Average Pairwise Environment Distance")
    plt.ylabel("Final PPO Performance")
    plt.title("Structured Environment Diversity vs PPO Performance")

    cbar = plt.colorbar(scatter)
    cbar.set_label("Scale")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        "ppo_structured_diversity_100envs.png",
        dpi=300
    )

    plt.show()