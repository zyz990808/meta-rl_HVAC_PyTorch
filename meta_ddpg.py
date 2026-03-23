import random
from copy import deepcopy
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.acts = np.zeros((size, act_dim), dtype=np.float32)
        self.rews = np.zeros((size, 1), dtype=np.float32)
        self.dones = np.zeros((size, 1), dtype=np.float32)

        self.max_size = size
        self.ptr = 0
        self.size = 0

    def add(self, obs, act, rew, next_obs, done):
        self.obs[self.ptr] = obs
        self.acts[self.ptr] = act
        self.rews[self.ptr] = rew
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)

        return {
            "obs": torch.tensor(self.obs[idx], dtype=torch.float32, device=DEVICE),
            "acts": torch.tensor(self.acts[idx], dtype=torch.float32, device=DEVICE),
            "rews": torch.tensor(self.rews[idx], dtype=torch.float32, device=DEVICE),
            "next_obs": torch.tensor(self.next_obs[idx], dtype=torch.float32, device=DEVICE),
            "dones": torch.tensor(self.dones[idx], dtype=torch.float32, device=DEVICE),
        }


class DDPGCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=128):
        super().__init__()
        self.q = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, obs, act):
        return self.q(torch.cat([obs, act], dim=-1))


def soft_update(target_net, source_net, tau):
    for tp, sp in zip(target_net.parameters(), source_net.parameters()):
        tp.data.copy_(tp.data * (1.0 - tau) + sp.data * tau)


def hard_update(target_net, source_net):
    target_net.load_state_dict(source_net.state_dict())


def select_deterministic_action(actor_critic, obs_tensor):
    with torch.no_grad():
        mean = actor_critic.actor(obs_tensor)
        action = torch.tanh(mean) * actor_critic.act_scale + actor_critic.act_bias
    return action


# =========================
# 🔥 modification here: add scaling
# =========================
def adapt_policy_ddpg(
    meta_model,
    env,
    adaptation_steps=400,
    replay_size=50000,
    batch_size=64,
    gamma=0.99,
    tau=0.005,
    actor_lr=1e-4,
    critic_lr=1e-3,
    start_steps=64,
    update_after=64,
    update_every=1,
    action_noise_std=0.05,
    hidden_size=128,
    max_episode_steps=1440,
    energy_scale=1.0,        # ⭐ NEW
    exceed_scale=1.0,        # ⭐ NEW
):

    # ⭐⭐⭐ important: sync reward scaling ⭐⭐⭐
    if hasattr(env, "update_scale"):
        env.update_scale(energy_scale, exceed_scale)

    adapted_model = deepcopy(meta_model).to(DEVICE)
    adapted_model.train()

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    critic = DDPGCritic(obs_dim, act_dim, hidden_size=hidden_size).to(DEVICE)
    critic_target = DDPGCritic(obs_dim, act_dim, hidden_size=hidden_size).to(DEVICE)
    actor_target = deepcopy(adapted_model).to(DEVICE)

    hard_update(critic_target, critic)
    hard_update(actor_target, adapted_model)

    actor_optimizer = optim.Adam(adapted_model.actor.parameters(), lr=actor_lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=critic_lr)

    replay_buffer = ReplayBuffer(obs_dim, act_dim, replay_size)

    obs = env.reset()
    episode_step = 0

    act_low = torch.tensor(env.action_space.low, dtype=torch.float32, device=DEVICE)
    act_high = torch.tensor(env.action_space.high, dtype=torch.float32, device=DEVICE)

    for t in range(adaptation_steps):
        episode_step += 1

        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)

        if t < start_steps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action_tensor = select_deterministic_action(adapted_model, obs_tensor)
                noise = torch.randn_like(action_tensor) * action_noise_std
                action_tensor = action_tensor + noise
                action_tensor = torch.max(torch.min(action_tensor, act_high), act_low)
                action = action_tensor.squeeze(0).cpu().numpy()

        next_obs, reward, done, _ = env.step(action)

        if episode_step >= max_episode_steps:
            done = True

        replay_buffer.add(obs, action, reward, next_obs, float(done))
        obs = next_obs

        if done:
            obs = env.reset()
            episode_step = 0

        if replay_buffer.size >= update_after and t % update_every == 0:
            batch = replay_buffer.sample(batch_size)

            with torch.no_grad():
                next_actions = select_deterministic_action(actor_target, batch["next_obs"])
                target_q = critic_target(batch["next_obs"], next_actions)
                y = batch["rews"] + gamma * (1.0 - batch["dones"]) * target_q

            q = critic(batch["obs"], batch["acts"])
            critic_loss = ((q - y) ** 2).mean()

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            pred_actions = select_deterministic_action(adapted_model, batch["obs"])
            actor_loss = -critic(batch["obs"], pred_actions).mean()

            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            soft_update(critic_target, critic, tau)
            soft_update(actor_target.actor, adapted_model.actor, tau)

    adapted_model.eval()
    return adapted_model


def reptile_update_actor(meta_model, adapted_models, step_size=0.1):

    if len(adapted_models) == 0:
        return

    with torch.no_grad():
        meta_actor_state = meta_model.actor.state_dict()

        avg_state = {}
        for key in meta_actor_state.keys():
            avg_state[key] = sum(
                model.actor.state_dict()[key] for model in adapted_models
            ) / len(adapted_models)

        new_state = {}
        for key in meta_actor_state.keys():
            new_state[key] = meta_actor_state[key] + step_size * (
                avg_state[key] - meta_actor_state[key]
            )

        meta_model.actor.load_state_dict(new_state)