import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from copy import deepcopy


# =========================
# Actor
# =========================
class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, act_low, act_high):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim),
            nn.Tanh()
        )

        self.act_low = torch.tensor(act_low, dtype=torch.float32)
        self.act_high = torch.tensor(act_high, dtype=torch.float32)

    def forward(self, x):
        x = self.model(x)
        return self.act_low + (x + 1) * 0.5 * (self.act_high - self.act_low)


# =========================
# Critic
# =========================
class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, obs, act):
        return self.model(torch.cat([obs, act], dim=-1))


# =========================
# Replay Buffer
# =========================
class ReplayBuffer:
    def __init__(self, size, obs_dim, act_dim):
        self.obs = np.zeros((size, obs_dim))
        self.next_obs = np.zeros((size, obs_dim))
        self.act = np.zeros((size, act_dim))
        self.rew = np.zeros((size, 1))
        self.done = np.zeros((size, 1))
        self.ptr = 0
        self.size = 0
        self.max_size = size

    def store(self, o, a, r, o2, d):
        self.obs[self.ptr] = o
        self.act[self.ptr] = a
        self.rew[self.ptr] = r
        self.next_obs[self.ptr] = o2
        self.done[self.ptr] = d
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return dict(
            obs=torch.tensor(self.obs[idx], dtype=torch.float32),
            act=torch.tensor(self.act[idx], dtype=torch.float32),
            rew=torch.tensor(self.rew[idx], dtype=torch.float32),
            next_obs=torch.tensor(self.next_obs[idx], dtype=torch.float32),
            done=torch.tensor(self.done[idx], dtype=torch.float32),
        )


# =========================
# DDPG main function (upgraded version)
# =========================
def ddpg_torch(env, ppo_actor=None, steps=5000, save_best=True):

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    act_low = env.action_space.low
    act_high = env.action_space.high

    actor = Actor(obs_dim, act_dim, act_low, act_high)
    critic = Critic(obs_dim, act_dim)

    target_actor = Actor(obs_dim, act_dim, act_low, act_high)
    target_critic = Critic(obs_dim, act_dim)

    # ===== meta initialization =====
    if ppo_actor is not None:
        print(">>> Using meta-policy to initialize actor")
        actor.model.load_state_dict(ppo_actor.state_dict())

    target_actor.load_state_dict(actor.state_dict())
    target_critic.load_state_dict(critic.state_dict())

    actor_opt = optim.Adam(actor.parameters(), lr=1e-3)
    critic_opt = optim.Adam(critic.parameters(), lr=1e-3)

    buffer = ReplayBuffer(100000, obs_dim, act_dim)

    # ===== track best model =====
    best_reward = -1e9
    best_actor = None

    o = env.reset()
    ep_return = 0
    episode_returns = []

    for t in range(steps):

        obs_t = torch.tensor(o, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            a = actor(obs_t).squeeze(0).numpy()

        # exploration
        a += 0.1 * np.random.randn(act_dim)
        a = np.clip(a, act_low, act_high)

        o2, r, d, _ = env.step(a)

        buffer.store(o, a, r, o2, d)

        o = o2
        ep_return += r

        # ===== update =====
        if buffer.size > 256:
            batch = buffer.sample(64)

            with torch.no_grad():
                next_a = target_actor(batch['next_obs'])
                target_q = batch['rew'] + 0.99 * (1 - batch['done']) * target_critic(batch['next_obs'], next_a)

            # critic
            q = critic(batch['obs'], batch['act'])
            critic_loss = ((q - target_q) ** 2).mean()

            critic_opt.zero_grad()
            critic_loss.backward()
            critic_opt.step()

            # actor
            actor_loss = -critic(batch['obs'], actor(batch['obs'])).mean()

            actor_opt.zero_grad()
            actor_loss.backward()
            actor_opt.step()

            # soft update
            for p, tp in zip(actor.parameters(), target_actor.parameters()):
                tp.data.copy_(0.995 * tp.data + 0.005 * p.data)

            for p, tp in zip(critic.parameters(), target_critic.parameters()):
                tp.data.copy_(0.995 * tp.data + 0.005 * p.data)

        # ===== end of episode =====
        if d:
            print(f"Episode done | Return: {ep_return:.2f}")
            episode_returns.append(ep_return)

            # ⭐ save best
            if ep_return > best_reward:
                best_reward = ep_return
                best_actor = deepcopy(actor)
                print(f">>> New BEST: {best_reward:.2f}")

            o = env.reset()
            ep_return = 0

        # ===== logging =====
        if (t + 1) % 500 == 0:
            print(f"[Step {t+1}] Current Episode Return: {ep_return:.2f}")

    # =========================
    # save model
    # =========================
    if save_best and best_actor is not None:
        torch.save(best_actor.state_dict(), "model/multi_env_meta/ddpg_actor_best.pth")
        print(">>> BEST actor saved")

    torch.save(actor.state_dict(), "model/multi_env_meta/ddpg_actor_final.pth")
    print(">>> FINAL actor saved")

    return best_reward, best_actor