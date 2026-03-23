# ⭐⭐⭐ FULL UPDATED VERSION ⭐⭐⭐

import os
import random
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from env import ContinuousBuildingControlEnvironment as BEnv
from meta_ddpg import adapt_policy_ddpg, reptile_update_actor


MAX_EPISODE_STEPS = 200

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TOTAL_META_UPDATES = 300
ROLLOUT_STEPS_PER_ENV = 512
NUM_EPOCHS = 10
MINIBATCH_SIZE = 256

GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_COEF = 0.2
VF_COEF = 0.5
ENT_COEF = 0.0
MAX_GRAD_NORM = 0.5
LR = 3e-4

HIDDEN_SIZE = 128

ADAPTATION_STEPS = 200
META_ACTOR_STEP_SIZE = 0.1

SAVE_EVERY = 25


def create_envs():
    envs = []
    params = pd.read_csv("env_param.csv")
    sampled = params.sample(n=5, random_state=42)

    for _, row in sampled.iterrows():
        env = BEnv(
            "weather_data_2013_to_2017_summer_pandas.csv",
            C_env=row["C_env"],
            C_air=row["C_air"],
            R_rc=row["R_rc"],
            R_oe=row["R_oe"],
            R_er=row["R_er"],
        )
        envs.append(env)

    return envs


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, act_low, act_high):
        super().__init__()

        self.actor = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN_SIZE),
            nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, act_dim),
        )

        self.log_std = nn.Parameter(torch.zeros(act_dim))

        self.critic = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN_SIZE),
            nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, 1),
        )

        self.register_buffer("act_low", torch.tensor(act_low, dtype=torch.float32))
        self.register_buffer("act_high", torch.tensor(act_high, dtype=torch.float32))
        self.register_buffer("act_scale", (self.act_high - self.act_low) / 2)
        self.register_buffer("act_bias", (self.act_high + self.act_low) / 2)

    def get_value(self, obs):
        return self.critic(obs).squeeze(-1)

    def get_action_and_value(self, obs):
        mean = self.actor(obs)
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(mean, std)
        z = dist.rsample()
        action = torch.tanh(z) * self.act_scale + self.act_bias

        logprob = dist.log_prob(z).sum(-1)
        value = self.get_value(obs)

        return action, logprob, value, dist.entropy().sum(-1)

    def evaluate_actions(self, obs, actions):
        mean = self.actor(obs)
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(mean, std)

        y = (actions - self.act_bias) / self.act_scale
        y = torch.clamp(y, -0.999999, 0.999999)
        z = 0.5 * torch.log((1 + y) / (1 - y))

        logprob = dist.log_prob(z).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.get_value(obs)

        return logprob, entropy, value


class RolloutBuffer:
    def __init__(self, total_steps, obs_dim, act_dim):
        self.total_steps = total_steps

        self.obs = torch.zeros((total_steps, obs_dim), device=DEVICE)
        self.actions = torch.zeros((total_steps, act_dim), device=DEVICE)
        self.logprobs = torch.zeros(total_steps, device=DEVICE)
        self.rewards = torch.zeros(total_steps, device=DEVICE)
        self.dones = torch.zeros(total_steps, device=DEVICE)
        self.values = torch.zeros(total_steps, device=DEVICE)

        self.advantages = torch.zeros(total_steps, device=DEVICE)
        self.returns = torch.zeros(total_steps, device=DEVICE)

        self.ptr = 0

    def add(self, obs, action, logprob, reward, done, value):
        i = self.ptr
        self.obs[i] = obs
        self.actions[i] = action
        self.logprobs[i] = logprob
        self.rewards[i] = reward
        self.dones[i] = done
        self.values[i] = value
        self.ptr += 1

    def compute_advantages(self, last_values):
        num_envs = len(last_values)
        chunk = self.total_steps // num_envs

        for env_idx in range(num_envs):
            start = env_idx * chunk
            end = start + chunk
            gae = 0.0

            for t in reversed(range(start, end)):
                next_value = last_values[env_idx] if t == end - 1 else self.values[t + 1]

                delta = self.rewards[t] + GAMMA * next_value * (1 - self.dones[t]) - self.values[t]
                gae = delta + GAMMA * GAE_LAMBDA * (1 - self.dones[t]) * gae
                self.advantages[t] = gae

        self.returns = self.advantages + self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)


def collect_rollout_for_env(policy_model, env, steps):
    obs = torch.tensor(env.reset(), dtype=torch.float32, device=DEVICE)

    traj = []
    energy_list = []
    exceed_list = []

    for _ in range(steps):

        with torch.no_grad():
            action, logprob, value, _ = policy_model.get_action_and_value(obs.unsqueeze(0))

        action = action.squeeze(0)
        logprob = logprob.squeeze(0)
        value = value.squeeze(0)

        next_obs, reward, done, info = env.step(action.cpu().numpy())

        traj.append((obs.clone(), action.clone(), logprob.clone(),
                     torch.tensor(reward, device=DEVICE),
                     torch.tensor(float(done), device=DEVICE),
                     value.clone()))

        energy_list.append(info["Energy"])
        exceed_list.append(info["Exceedance"])

        obs = torch.tensor(next_obs, dtype=torch.float32, device=DEVICE)

        if done:
            obs = torch.tensor(env.reset(), dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        last_value = policy_model.get_value(obs.unsqueeze(0)).squeeze(0)

    return traj, last_value, energy_list, exceed_list


def train():

    save_dir = "model/multi_env_meta"
    os.makedirs(save_dir, exist_ok=True)

    set_seed(SEED)
    envs = create_envs()

    obs_dim = envs[0].observation_space.shape[0]
    act_dim = envs[0].action_space.shape[0]

    model = ActorCritic(obs_dim, act_dim,
                        envs[0].action_space.low,
                        envs[0].action_space.high).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR)

    buffer = RolloutBuffer(len(envs) * ROLLOUT_STEPS_PER_ENV, obs_dim, act_dim)

    # ===== scaling factors =====
    energy_scale = 1.0
    exceed_scale = 1.0

    for update in range(1, TOTAL_META_UPDATES + 1):

        buffer.ptr = 0
        adapted_models = []
        bootstrap_values = []

        all_energy = []
        all_exceed = []

        # apply scaling to all environments
        for env in envs:
            env.update_scale(energy_scale, exceed_scale)

        for env in envs:

            task_model = deepcopy(model).to(DEVICE)

            adapted_model = adapt_policy_ddpg(
                task_model,
                env,
                ADAPTATION_STEPS,
                energy_scale=energy_scale,
                exceed_scale=exceed_scale,
            )
            adapted_models.append(adapted_model)

            traj, last_value, energy_list, exceed_list = collect_rollout_for_env(
                adapted_model, env, ROLLOUT_STEPS_PER_ENV
            )

            bootstrap_values.append(last_value)

            all_energy.extend(energy_list)
            all_exceed.extend(exceed_list)

            for item in traj:
                buffer.add(*item)

        # ===== PPO update =====
        buffer.compute_advantages(bootstrap_values)

        for _ in range(NUM_EPOCHS):
            idx = torch.randperm(buffer.ptr, device=DEVICE)

            for start in range(0, buffer.ptr, MINIBATCH_SIZE):
                mb = idx[start:start + MINIBATCH_SIZE]

                logp, entropy, value = model.evaluate_actions(buffer.obs[mb], buffer.actions[mb])

                ratio = (logp - buffer.logprobs[mb]).exp()

                loss = torch.max(
                    -buffer.advantages[mb] * ratio,
                    -buffer.advantages[mb] * torch.clamp(ratio, 1 - CLIP_COEF, 1 + CLIP_COEF)
                ).mean()

                loss += VF_COEF * ((value - buffer.returns[mb]) ** 2).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        reptile_update_actor(model, adapted_models, META_ACTOR_STEP_SIZE)

        # ===== update scaling =====
        if len(all_energy) > 0:
            m_E = np.percentile(all_energy, 95)
            m_X = np.percentile(all_exceed, 95)

            energy_scale = 0.9 * energy_scale + 0.1 * m_E
            exceed_scale = 0.9 * exceed_scale + 0.1 * m_X

            print(f"[Scale] E={energy_scale:.2f}, X={exceed_scale:.2f}")

        print(f"[Meta Update {update}] done")

    torch.save(model.state_dict(), os.path.join(save_dir, "ppo_multi_env_meta_final.pth"))
    print("Final model saved")


if __name__ == "__main__":
    train()