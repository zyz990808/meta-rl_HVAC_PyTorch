# ddpg_meta.py

import os
import numpy as np
import torch
import torch.optim as optim

from ddpg_torch import Actor, Critic, ReplayBuffer


SAVE_DIR = "model"
os.makedirs(SAVE_DIR, exist_ok=True)


# =========================
# Copy PPO Actor → DDPG Actor
# =========================
def copy_ppo_to_ddpg(ppo_actor, ddpg_actor):
    ppo_layers = [m for m in ppo_actor.modules() if isinstance(m, torch.nn.Linear)]
    ddpg_layers = [m for m in ddpg_actor.modules() if isinstance(m, torch.nn.Linear)]

    for src, dst in zip(ppo_layers, ddpg_layers):
        dst.weight.data.copy_(src.weight.data)
        dst.bias.data.copy_(src.bias.data)


# =========================
# Evaluate final policy
# =========================
def evaluate_actor(env, actor, device="cpu", steps=720):
    obs = env.reset()
    total_reward = 0.0

    for _ in range(steps):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            action = actor(obs_t).squeeze(0).cpu().numpy()

        obs, reward, done, _ = env.step(action)
        total_reward += reward

        if done:
            break

    return total_reward


# =========================
# DDPG Adaptation
# =========================
def ddpg_adapt(
    env,
    ppo_actor=None,
    device="cpu",
    steps=100000,
    batch_size=64,
    gamma=0.99,
    tau=0.005,
    actor_lr=1e-4,
    critic_lr=1e-4,
    noise_std=0.1,
):

    # ----- Init networks -----
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))

    act_low = env.action_space.low.astype(np.float32)
    act_high = env.action_space.high.astype(np.float32)

    actor = Actor(obs_dim, act_dim, act_low, act_high).to(device)
    critic = Critic(obs_dim, act_dim).to(device)

    target_actor = Actor(obs_dim, act_dim, act_low, act_high).to(device)
    target_critic = Critic(obs_dim, act_dim).to(device)

    # ----- Init actor from PPO -----
    if ppo_actor is not None:
        copy_ppo_to_ddpg(ppo_actor, actor)

    # =========================
    # No Q warm-start
    # Always use random critic
    # =========================
    print(">> Using random Q")

    target_actor.load_state_dict(actor.state_dict())
    target_critic.load_state_dict(critic.state_dict())

    actor_opt = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_opt = optim.Adam(critic.parameters(), lr=critic_lr)

    buffer = ReplayBuffer(100000, obs_dim, act_dim)

    # =========================
    # Warm-up setting
    # 30% warm up steps
    # =========================
    warmup_steps = int(0.3 * steps)

    obs = env.reset()

    # =========================
    # Training loop
    # =========================
    for t in range(steps):

        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            action = actor(obs_t).squeeze(0).cpu().numpy()

        action += noise_std * np.random.randn(act_dim)
        action = np.clip(action, act_low, act_high)

        next_obs, reward, done, _ = env.step(action)

        buffer.store(obs, action, reward, next_obs, done)

        obs = next_obs

        if buffer.size > batch_size:
            batch = buffer.sample(batch_size)

            o = batch["obs"].to(device)
            a = batch["act"].to(device)
            r = batch["rew"].to(device)
            o2 = batch["next_obs"].to(device)
            d = batch["done"].to(device)

            # ----- Critic update -----
            with torch.no_grad():
                a2 = target_actor(o2)
                q_target = r + gamma * (1 - d) * target_critic(o2, a2)

            q = critic(o, a)
            loss_q = ((q - q_target) ** 2).mean()

            critic_opt.zero_grad()
            loss_q.backward()
            critic_opt.step()

            # ----- Actor update -----
            if t > warmup_steps:
                loss_pi = -critic(o, actor(o)).mean()

                actor_opt.zero_grad()
                loss_pi.backward()
                actor_opt.step()

            # ----- Soft update -----
            for p, tp in zip(actor.parameters(), target_actor.parameters()):
                tp.data.copy_(tau * p.data + (1 - tau) * tp.data)

            for p, tp in zip(critic.parameters(), target_critic.parameters()):
                tp.data.copy_(tau * p.data + (1 - tau) * tp.data)

        if done:
            obs = env.reset()

    # =========================
    # Final evaluation
    # =========================
    final_return = evaluate_actor(env, actor, device)

    return actor, final_return