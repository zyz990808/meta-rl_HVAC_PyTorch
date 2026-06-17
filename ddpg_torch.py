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
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, act_dim),
            nn.Tanh(),
        )

        self.register_buffer(
            "act_low",
            torch.tensor(act_low, dtype=torch.float32),
        )

        self.register_buffer(
            "act_high",
            torch.tensor(act_high, dtype=torch.float32),
        )

    def forward(self, x):
        x = self.model(x)
        return self.act_low + (x + 1.0) * 0.5 * (
            self.act_high - self.act_low
        )


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
            nn.Linear(128, 1),
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

        return {
            "obs": torch.tensor(self.obs[idx], dtype=torch.float32),
            "act": torch.tensor(self.act[idx], dtype=torch.float32),
            "rew": torch.tensor(self.rew[idx], dtype=torch.float32),
            "next_obs": torch.tensor(self.next_obs[idx], dtype=torch.float32),
            "done": torch.tensor(self.done[idx], dtype=torch.float32),
        }


# =========================
# Exceedance helpers
# =========================
def get_bounds(env):
    lb = getattr(env, "lb_set", 22.0)
    ub = getattr(env, "ub_set", 24.0)
    return lb, ub


def compute_step_exceedance(obs, env):
    lb, ub = get_bounds(env)

    temp = obs[0]

    dt = getattr(env, "dt", 1800.0)
    hours = dt / 3600.0

    exceed = max(0.0, lb - temp) + max(0.0, temp - ub)

    return exceed * hours


# =========================
# Copy PPO Actor -> DDPG Actor
# =========================
def copy_ppo_to_ddpg(ppo_actor, ddpg_actor):
    ppo_layers = [
        m for m in ppo_actor.modules()
        if isinstance(m, nn.Linear)
    ]

    ddpg_layers = [
        m for m in ddpg_actor.modules()
        if isinstance(m, nn.Linear)
    ]

    for src, dst in zip(ppo_layers, ddpg_layers):
        dst.weight.data.copy_(src.weight.data)
        dst.bias.data.copy_(src.bias.data)


# =========================
# DDPG Main
# PPO Actor + Optional Pretrained Q + Warm-up
# Best-point restart, no early stop
# =========================
def ddpg_torch(
    env,
    ppo_actor=None,
    steps=50000,
    warmup_steps=20000,
    batch_size=64,
    update_after=256,
    gamma=0.99,
    tau=0.005,
    actor_lr=1e-5,
    critic_lr=1e-5,
    noise_std=0.05,
    actor_step_scale=0.03,
    safety_return_threshold=-10.0,
    pretrained_critic=None,
    pretrained_target_critic=None,
    tolerance_steps=5,
):

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    act_low = env.action_space.low
    act_high = env.action_space.high

    # =========================
    # Actor
    # =========================
    actor = Actor(
        obs_dim,
        act_dim,
        act_low,
        act_high,
    )

    # =========================
    # Critic
    # =========================
    if pretrained_critic is not None:
        print(">>> Using pretrained/shared Q")
        critic = pretrained_critic
    else:
        print(">>> Using random Q")
        critic = Critic(
            obs_dim,
            act_dim,
        )

    if pretrained_target_critic is not None:
        target_critic = pretrained_target_critic
    else:
        target_critic = Critic(
            obs_dim,
            act_dim,
        )
        target_critic.load_state_dict(
            critic.state_dict()
        )

    # =========================
    # Target Actor
    # =========================
    target_actor = Actor(
        obs_dim,
        act_dim,
        act_low,
        act_high,
    )

    # =========================
    # Initialize actor from PPO
    # =========================
    if ppo_actor is not None:
        print(">>> Initialize DDPG actor from PPO")
        copy_ppo_to_ddpg(
            ppo_actor,
            actor,
        )

    target_actor.load_state_dict(
        actor.state_dict()
    )

    # =========================
    # Optimizers
    # =========================
    actor_opt = optim.Adam(
        actor.parameters(),
        lr=actor_lr,
    )

    critic_opt = optim.Adam(
        critic.parameters(),
        lr=critic_lr,
    )

    # =========================
    # Replay buffer
    # =========================
    buffer = ReplayBuffer(
        100000,
        obs_dim,
        act_dim,
    )

    # =========================
    # Tracking
    # =========================
    best_actor = deepcopy(actor)
    best_reward = -1e9

    last_records = []

    o = env.reset()

    ep_return = 0.0
    ep_exceedance = 0.0

    episode_returns = []
    episode_exceedances = []

    bad_episode_counter = 0
    restart_count = 0

    print(f">>> Warm-up steps: {warmup_steps}")
    print(f">>> Safety return threshold: {safety_return_threshold}")
    print(f">>> Tolerance steps: {tolerance_steps}")
    print(">>> Best-point restart enabled")
    print(">>> Training will continue until total steps are finished")

    # =========================
    # Training loop
    # =========================
    for t in range(steps):

        obs_t = torch.tensor(
            o,
            dtype=torch.float32,
        ).unsqueeze(0)

        with torch.no_grad():
            a = actor(obs_t).squeeze(0).numpy()

        # Exploration noise during adaptation
        a += noise_std * np.random.randn(act_dim)
        a = np.clip(
            a,
            act_low,
            act_high,
        )

        o2, r, d, _ = env.step(a)

        buffer.store(
            o,
            a,
            r,
            o2,
            d,
        )

        o = o2
        ep_return += r
        ep_exceedance += compute_step_exceedance(
            o2,
            env,
        )

        # =========================
        # Network update
        # =========================
        if buffer.size > update_after:

            batch = buffer.sample(
                batch_size
            )

            with torch.no_grad():
                next_a = target_actor(
                    batch["next_obs"]
                )

                target_q = batch["rew"] + gamma * (
                    1 - batch["done"]
                ) * target_critic(
                    batch["next_obs"],
                    next_a,
                )

            # ----- critic update -----
            q = critic(
                batch["obs"],
                batch["act"],
            )

            critic_loss = ((q - target_q) ** 2).mean()

            critic_opt.zero_grad()
            critic_loss.backward()
            critic_opt.step()

            # ----- actor update after warm-up -----
            if t > warmup_steps:
                old_actor = deepcopy(actor)

                actor_loss = -critic(
                    batch["obs"],
                    actor(batch["obs"]),
                ).mean()

                actor_opt.zero_grad()
                actor_loss.backward()
                actor_opt.step()

                # Small actor step for stability
                for p, old_p in zip(
                    actor.parameters(),
                    old_actor.parameters(),
                ):
                    p.data.copy_(
                        old_p.data
                        + actor_step_scale * (p.data - old_p.data)
                    )

            # ----- soft update target actor -----
            for p, tp in zip(
                actor.parameters(),
                target_actor.parameters(),
            ):
                tp.data.copy_(
                    (1 - tau) * tp.data
                    + tau * p.data
                )

            # ----- soft update target critic -----
            for p, tp in zip(
                critic.parameters(),
                target_critic.parameters(),
            ):
                tp.data.copy_(
                    (1 - tau) * tp.data
                    + tau * p.data
                )

        # =========================
        # Episode end
        # =========================
        if d:
            print(
                f"[DDPG] Episode done | "
                f"Return: {ep_return:.2f} | "
                f"Exceedance: {ep_exceedance:.2f}"
            )

            episode_returns.append(
                ep_return
            )

            episode_exceedances.append(
                ep_exceedance
            )

            # =========================
            # Update best point
            # =========================
            if ep_return > best_reward:
                best_reward = ep_return
                best_actor = deepcopy(actor)
                bad_episode_counter = 0

                print(
                    f">>> New best actor found | "
                    f"best_reward = {best_reward:.2f}"
                )

            else:
                if t > warmup_steps:
                    bad_episode_counter += 1

                    print(
                        f">>> No improvement over best_reward "
                        f"({ep_return:.2f} <= {best_reward:.2f})"
                    )

                    print(
                        f">>> Bad episode counter: "
                        f"{bad_episode_counter}/{tolerance_steps}"
                    )

            # =========================
            # Safety guard
            # =========================
            if (
                t > warmup_steps
                and ep_return < safety_return_threshold
            ):
                print(
                    f">>> Safety guard triggered "
                    f"(return {ep_return:.2f} < "
                    f"{safety_return_threshold:.2f})"
                )
                print(">>> Restore best actor and continue")

                actor.load_state_dict(
                    best_actor.state_dict()
                )

                target_actor.load_state_dict(
                    actor.state_dict()
                )

                bad_episode_counter = 0
                restart_count += 1

            # =========================
            # Best-point restart
            # =========================
            if (
                t > warmup_steps
                and bad_episode_counter >= tolerance_steps
            ):
                print(
                    ">>> Tolerance exceeded after best point"
                )

                print(
                    ">>> Restart from best actor and continue training"
                )

                actor.load_state_dict(
                    best_actor.state_dict()
                )

                target_actor.load_state_dict(
                    actor.state_dict()
                )

                bad_episode_counter = 0
                restart_count += 1

            # =========================
            # Record last episodes after warm-up
            # =========================
            if t > warmup_steps:
                last_records.append(
                    {
                        "actor": deepcopy(actor),
                        "return": ep_return,
                        "exceedance": ep_exceedance,
                    }
                )

                if len(last_records) > 15:
                    last_records.pop(0)

            # Reset episode
            o = env.reset()
            ep_return = 0.0
            ep_exceedance = 0.0

    # =========================
    # Select from last episodes
    # =========================
    if len(last_records) > 0:
        last_best_record = max(
            last_records,
            key=lambda x: x["return"],
        )

        last_min_exceed_record = min(
            last_records,
            key=lambda x: x["exceedance"],
        )

        last_best_actor = last_best_record["actor"]
        last_min_exceed_actor = last_min_exceed_record["actor"]

        last_best_reward = last_best_record["return"]
        last_min_exceedance = last_min_exceed_record["exceedance"]

    else:
        last_best_actor = deepcopy(actor)
        last_min_exceed_actor = deepcopy(actor)

        last_best_reward = None
        last_min_exceedance = None

    # =========================
    # Return
    # =========================
    return {
        "best_actor": best_actor,
        "final_actor": actor,

        "last_best_actor": last_best_actor,
        "last_min_exceed_actor": last_min_exceed_actor,

        "critic": critic,
        "target_critic": target_critic,

        "best_reward": best_reward,
        "last_best_reward": last_best_reward,
        "last_min_exceedance": last_min_exceedance,

        "episode_returns": episode_returns,
        "episode_exceedances": episode_exceedances,

        "restart_count": restart_count,
        "tolerance_steps": tolerance_steps,
    }