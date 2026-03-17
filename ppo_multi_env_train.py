import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

from env import ContinuousBuildingControlEnvironment as BEnv

MAX_EPISODE_STEPS = 1440

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TOTAL_TIMESTEPS = 1000000
ROLLOUT_STEPS = 2048
NUM_EPOCHS = 10
MINIBATCH_SIZE = 256

GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_COEF = 0.2
VF_COEF = 0.5
ENT_COEF = 0.0
MAX_GRAD_NORM = 0.5
LR = 3e-4
TARGET_KL = 0.02

HIDDEN_SIZE = 128


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
            nn.Linear(HIDDEN_SIZE, act_dim)
        )

        self.log_std = nn.Parameter(torch.zeros(act_dim))

        self.critic = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN_SIZE),
            nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, 1)
        )

        self.register_buffer("act_low", torch.tensor(act_low))
        self.register_buffer("act_high", torch.tensor(act_high))

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

    def __init__(self, obs_dim, act_dim):

        self.obs = torch.zeros((ROLLOUT_STEPS, obs_dim)).to(DEVICE)
        self.actions = torch.zeros((ROLLOUT_STEPS, act_dim)).to(DEVICE)
        self.logprobs = torch.zeros(ROLLOUT_STEPS).to(DEVICE)
        self.rewards = torch.zeros(ROLLOUT_STEPS).to(DEVICE)
        self.dones = torch.zeros(ROLLOUT_STEPS).to(DEVICE)
        self.values = torch.zeros(ROLLOUT_STEPS).to(DEVICE)

        self.advantages = torch.zeros(ROLLOUT_STEPS).to(DEVICE)
        self.returns = torch.zeros(ROLLOUT_STEPS).to(DEVICE)

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

    def compute_advantages(self, last_value):

        gae = 0

        for t in reversed(range(ROLLOUT_STEPS)):

            next_value = last_value if t == ROLLOUT_STEPS - 1 else self.values[t + 1]

            delta = self.rewards[t] + GAMMA * next_value * (1 - self.dones[t]) - self.values[t]

            gae = delta + GAMMA * GAE_LAMBDA * (1 - self.dones[t]) * gae

            self.advantages[t] = gae

        self.returns = self.advantages + self.values

        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)


### def create_envs():

###    envs = []

###    params = [
###        (3.2e6, 3.5e5, 0.007, 0.027, 0.0036),
###        (3.0e6, 3.4e5, 0.008, 0.023, 0.0030),
###        (3.4e6, 3.3e5, 0.005, 0.029, 0.0041),
###        (3.1e6, 3.2e5, 0.004, 0.024, 0.0031),
###        (3.3e6, 3.7e5, 0.006, 0.026, 0.0039),
###    ]

###    for p in params:

###        env = BEnv(
###            "weather_data_2013_to_2017_summer_pandas.csv",
###            C_env=p[0],
###            C_air=p[1],
###            R_rc=p[2],
###            R_oe=p[3],
###            R_er=p[4],
###        )

###        envs.append(env)

###    return envs


def train():

    save_dir = "model/multi_env"
    os.makedirs(save_dir, exist_ok=True)

    set_seed(SEED)

    envs = create_envs()

    env = random.choice(envs)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    act_low = env.action_space.low
    act_high = env.action_space.high

    model = ActorCritic(obs_dim, act_dim, act_low, act_high).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR)

    buffer = RolloutBuffer(obs_dim, act_dim)

    obs = torch.tensor(env.reset()).float().to(DEVICE)

    num_updates = TOTAL_TIMESTEPS // ROLLOUT_STEPS

    global_step = 0

    episode_return = 0
    episode_returns = []
    episodes = 0

    for update in range(1, num_updates + 1):

        buffer.ptr = 0

        episode_step = 0

        for step in range(ROLLOUT_STEPS):

            episode_step += 1

            global_step += 1

            with torch.no_grad():
                action, logprob, value, _ = model.get_action_and_value(obs.unsqueeze(0))

            action = action.squeeze(0)

            next_obs, reward, done, info = env.step(action.cpu().numpy())

            if episode_step >= MAX_EPISODE_STEPS:
                done = True

            buffer.add(
                obs,
                action,
                logprob,
                torch.tensor(reward).to(DEVICE),
                torch.tensor(done).to(DEVICE),
                value
            )

            episode_return += reward

            obs = torch.tensor(next_obs).float().to(DEVICE)

            if done:
                episode_step = 0

                episode_returns.append(episode_return)
                episodes += 1
                episode_return = 0

                env = random.choice(envs)

                obs = torch.tensor(env.reset()).float().to(DEVICE)

        with torch.no_grad():
            last_value = model.get_value(obs.unsqueeze(0))

        buffer.compute_advantages(last_value)

        for epoch in range(NUM_EPOCHS):

            idx = torch.randperm(ROLLOUT_STEPS)

            for start in range(0, ROLLOUT_STEPS, MINIBATCH_SIZE):

                mb = idx[start:start + MINIBATCH_SIZE]

                obs_b = buffer.obs[mb]
                act_b = buffer.actions[mb]
                logp_old = buffer.logprobs[mb]
                adv_b = buffer.advantages[mb]
                ret_b = buffer.returns[mb]

                logp, entropy, value = model.evaluate_actions(obs_b, act_b)

                ratio = (logp - logp_old).exp()

                pg_loss1 = -adv_b * ratio
                pg_loss2 = -adv_b * torch.clamp(ratio, 1 - CLIP_COEF, 1 + CLIP_COEF)

                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = ((value - ret_b) ** 2).mean()

                loss = pg_loss + VF_COEF * v_loss

                optimizer.zero_grad()
                loss.backward()

                nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

                optimizer.step()

        avg10 = np.mean(episode_returns[-10:]) if len(episode_returns) > 0 else 0

        print(f"[Update {update}] steps={global_step} episodes={episodes} avgReturn={avg10}")

        if update % 50 == 0:
            model_path = os.path.join(save_dir, f"ppo_multi_env_{update}.pth")
            torch.save(model.state_dict(), model_path)
            print(f"[Checkpoint] Saved model at update {update}")

    model_path = os.path.join(save_dir, "ppo_multi_env.pth")
    torch.save(model.state_dict(), model_path)

    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    train()