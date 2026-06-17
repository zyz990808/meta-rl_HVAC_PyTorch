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
            nn.Tanh()
        )

        self.register_buffer(
            "act_low",
            torch.tensor(act_low, dtype=torch.float32)
        )

        self.register_buffer(
            "act_high",
            torch.tensor(act_high, dtype=torch.float32)
        )

    def forward(self, x):
        x = self.model(x)
        return self.act_low + (x + 1.0) * 0.5 * (self.act_high - self.act_low)


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
        x = torch.cat(
            [obs, act],
            dim=-1
        )
        return self.model(x)


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
        self.size = min(
            self.size + 1,
            self.max_size
        )

    def sample(self, batch_size):
        idx = np.random.randint(
            0,
            self.size,
            size=batch_size
        )

        return dict(
            obs=torch.tensor(self.obs[idx], dtype=torch.float32),
            act=torch.tensor(self.act[idx], dtype=torch.float32),
            rew=torch.tensor(self.rew[idx], dtype=torch.float32),
            next_obs=torch.tensor(self.next_obs[idx], dtype=torch.float32),
            done=torch.tensor(self.done[idx], dtype=torch.float32),
        )


# =========================
# Exceedance helpers
# =========================
def get_bounds(env):
    lb = getattr(env, "lb_set", 22.0)
    ub = getattr(env, "ub_set", 24.0)
    return lb, ub


def compute_step_exceedance(obs, env):
    lb, ub = get_bounds(env)

    # obs[0] is T_air
    temp = obs[0]

    dt = getattr(env, "dt", 1800.0)
    hours = dt / 3600.0

    exceed = max(0.0, lb - temp) + max(0.0, temp - ub)

    return exceed * hours


# =========================
# Conservative interpolation
# =========================
def interpolate_actor_(actor, old_actor, candidate_actor, alpha=0.1):
    """
    actor <- (1 - alpha) * old_actor + alpha * candidate_actor
    """

    for p, old_p, cand_p in zip(
        actor.parameters(),
        old_actor.parameters(),
        candidate_actor.parameters()
    ):
        p.data.copy_(
            (1.0 - alpha) * old_p.data + alpha * cand_p.data
        )


# =========================
# TD3 Main
# PPO initialization + Random Twin Critics + Warm-up
# =========================
def td3_torch(
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
    target_noise_std=0.1,
    target_noise_clip=0.2,
    policy_delay=2,
    actor_step_scale=0.03,
    bad_update_alpha=0.1,
    patience=10,
    tolerance=2.0,
):

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    act_low = env.action_space.low
    act_high = env.action_space.high

    act_low_t = torch.tensor(
        act_low,
        dtype=torch.float32
    )

    act_high_t = torch.tensor(
        act_high,
        dtype=torch.float32
    )

    # =========================
    # Actor
    # =========================
    actor = Actor(
        obs_dim,
        act_dim,
        act_low,
        act_high
    )

    target_actor = Actor(
        obs_dim,
        act_dim,
        act_low,
        act_high
    )

    # =========================
    # Twin Critics
    # =========================
    print(">>> Using random twin critics")

    critic1 = Critic(
        obs_dim,
        act_dim
    )

    critic2 = Critic(
        obs_dim,
        act_dim
    )

    target_critic1 = Critic(
        obs_dim,
        act_dim
    )

    target_critic2 = Critic(
        obs_dim,
        act_dim
    )

    target_critic1.load_state_dict(
        critic1.state_dict()
    )

    target_critic2.load_state_dict(
        critic2.state_dict()
    )

    # =========================
    # Initialize actor from PPO
    # =========================
    if ppo_actor is not None:
        print(">>> Initialize TD3 actor from PPO")

        ppo_layers = [
            m for m in ppo_actor.modules()
            if isinstance(m, nn.Linear)
        ]

        td3_layers = [
            m for m in actor.modules()
            if isinstance(m, nn.Linear)
        ]

        for src, dst in zip(
            ppo_layers,
            td3_layers
        ):
            dst.weight.data.copy_(
                src.weight.data
            )

            dst.bias.data.copy_(
                src.bias.data
            )

    target_actor.load_state_dict(
        actor.state_dict()
    )

    # =========================
    # Optimizers
    # =========================
    actor_opt = optim.Adam(
        actor.parameters(),
        lr=actor_lr
    )

    critic1_opt = optim.Adam(
        critic1.parameters(),
        lr=critic_lr
    )

    critic2_opt = optim.Adam(
        critic2.parameters(),
        lr=critic_lr
    )

    buffer = ReplayBuffer(
        100000,
        obs_dim,
        act_dim
    )

    # =========================
    # Tracking
    # =========================
    best_actor = deepcopy(actor)
    best_reward = -1e9

    bad_episode_count = 0

    last_records = []

    o = env.reset()
    ep_return = 0.0
    ep_exceedance = 0.0

    episode_returns = []
    episode_exceedances = []

    prev_episode_return = None
    episode_start_actor = deepcopy(actor)

    print(f">>> Warm-up steps: {warmup_steps}")
    print(f">>> Policy delay: {policy_delay}")
    print(f">>> Target noise std: {target_noise_std}")
    print(f">>> Target noise clip: {target_noise_clip}")
    print(f">>> Patience: {patience}")
    print(f">>> Tolerance: {tolerance}")

    # =========================
    # Training loop
    # =========================
    for t in range(steps):

        obs_t = torch.tensor(
            o,
            dtype=torch.float32
        ).unsqueeze(0)

        with torch.no_grad():
            a = actor(obs_t).squeeze(0).numpy()

        # exploration noise
        a += noise_std * np.random.randn(act_dim)
        a = np.clip(
            a,
            act_low,
            act_high
        )

        o2, r, d, _ = env.step(a)

        buffer.store(
            o,
            a,
            r,
            o2,
            d
        )

        o = o2
        ep_return += r
        ep_exceedance += compute_step_exceedance(
            o2,
            env
        )

        # =========================
        # TD3 update
        # =========================
        if buffer.size > update_after:

            batch = buffer.sample(batch_size)

            # -------------------------
            # Target action smoothing
            # -------------------------
            with torch.no_grad():

                next_a = target_actor(
                    batch["next_obs"]
                )

                target_noise = target_noise_std * torch.randn_like(next_a)

                target_noise = torch.clamp(
                    target_noise,
                    -target_noise_clip,
                    target_noise_clip
                )

                next_a = next_a + target_noise

                next_a = torch.max(
                    torch.min(next_a, act_high_t),
                    act_low_t
                )

                target_q1 = target_critic1(
                    batch["next_obs"],
                    next_a
                )

                target_q2 = target_critic2(
                    batch["next_obs"],
                    next_a
                )

                target_q = torch.min(
                    target_q1,
                    target_q2
                )

                target_q = batch["rew"] + gamma * (1.0 - batch["done"]) * target_q

            # -------------------------
            # Critic 1 update
            # -------------------------
            q1 = critic1(
                batch["obs"],
                batch["act"]
            )

            critic1_loss = ((q1 - target_q) ** 2).mean()

            critic1_opt.zero_grad()
            critic1_loss.backward()
            critic1_opt.step()

            # -------------------------
            # Critic 2 update
            # -------------------------
            q2 = critic2(
                batch["obs"],
                batch["act"]
            )

            critic2_loss = ((q2 - target_q) ** 2).mean()

            critic2_opt.zero_grad()
            critic2_loss.backward()
            critic2_opt.step()

            # -------------------------
            # Delayed actor update
            # -------------------------
            if t > warmup_steps and t % policy_delay == 0:

                old_actor = deepcopy(actor)

                actor_loss = -critic1(
                    batch["obs"],
                    actor(batch["obs"])
                ).mean()

                actor_opt.zero_grad()
                actor_loss.backward()
                actor_opt.step()

                # small actor step for stability
                for p, old_p in zip(
                    actor.parameters(),
                    old_actor.parameters()
                ):
                    p.data.copy_(
                        old_p.data + actor_step_scale * (p.data - old_p.data)
                    )

                # soft update target actor only when actor is updated
                for p, tp in zip(
                    actor.parameters(),
                    target_actor.parameters()
                ):
                    tp.data.copy_(
                        (1.0 - tau) * tp.data + tau * p.data
                    )

            # -------------------------
            # Soft update target critics
            # -------------------------
            for p, tp in zip(
                critic1.parameters(),
                target_critic1.parameters()
            ):
                tp.data.copy_(
                    (1.0 - tau) * tp.data + tau * p.data
                )

            for p, tp in zip(
                critic2.parameters(),
                target_critic2.parameters()
            ):
                tp.data.copy_(
                    (1.0 - tau) * tp.data + tau * p.data
                )

        # =========================
        # Episode end
        # =========================
        if d:
            print(
                f"[TD3] Episode done | Return: {ep_return:.2f} | "
                f"Exceedance: {ep_exceedance:.2f}"
            )

            # =================================================
            # Episode-level conservative interpolation
            # =================================================
            if t > warmup_steps and prev_episode_return is not None:

                candidate_actor = deepcopy(actor)

                if ep_return >= prev_episode_return:
                    print(
                        f">>> Accept full policy update "
                        f"(current return {ep_return:.2f} >= previous {prev_episode_return:.2f})"
                    )

                    actor.load_state_dict(
                        candidate_actor.state_dict()
                    )

                else:
                    print(
                        f">>> Bad update detected "
                        f"(current return {ep_return:.2f} < previous {prev_episode_return:.2f})"
                    )

                    print(
                        f">>> Apply conservative update: "
                        f"theta = {(1.0 - bad_update_alpha):.1f} old + "
                        f"{bad_update_alpha:.1f} candidate"
                    )

                    interpolate_actor_(
                        actor,
                        episode_start_actor,
                        candidate_actor,
                        alpha=bad_update_alpha
                    )

                target_actor.load_state_dict(
                    actor.state_dict()
                )

            episode_returns.append(ep_return)
            episode_exceedances.append(ep_exceedance)

            # =========================
            # Track best actor
            # =========================
            if ep_return > best_reward:

                best_reward = ep_return
                best_actor = deepcopy(actor)
                bad_episode_count = 0

                print(
                    f">>> New best actor saved "
                    f"(best return = {best_reward:.2f})"
                )

            else:

                if ep_return < best_reward - tolerance:
                    bad_episode_count += 1

                    print(
                        f">>> Bad episode count: {bad_episode_count}/{patience} "
                        f"(return {ep_return:.2f} < best {best_reward:.2f} - tolerance {tolerance:.2f})"
                    )

                else:
                    bad_episode_count = max(
                        0,
                        bad_episode_count - 1
                    )

                    print(
                        f">>> Within tolerance region. "
                        f"Bad episode count reduced to {bad_episode_count}/{patience}"
                    )

            # =========================
            # Patience rollback
            # =========================
            if t > warmup_steps and bad_episode_count >= patience:

                print(
                    f">>> Patience rollback triggered "
                    f"(bad episodes = {bad_episode_count})"
                )

                print(
                    f">>> Restore best actor "
                    f"(best return = {best_reward:.2f})"
                )

                actor.load_state_dict(
                    best_actor.state_dict()
                )

                target_actor.load_state_dict(
                    actor.state_dict()
                )

                bad_episode_count = 0

            # =========================
            # Only record episodes AFTER warm-up
            # =========================
            if t > warmup_steps:

                last_records.append({
                    "actor": deepcopy(actor),
                    "return": ep_return,
                    "exceedance": ep_exceedance,
                })

                if len(last_records) > 15:
                    last_records.pop(0)

            prev_episode_return = ep_return

            o = env.reset()
            ep_return = 0.0
            ep_exceedance = 0.0

            episode_start_actor = deepcopy(actor)

    # =========================
    # Select from last episodes
    # =========================
    if len(last_records) > 0:
        last_best_record = max(
            last_records,
            key=lambda x: x["return"]
        )

        last_min_exceed_record = min(
            last_records,
            key=lambda x: x["exceedance"]
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

    return {
        "best_actor": best_actor,
        "final_actor": actor,

        "last_best_actor": last_best_actor,
        "last_min_exceed_actor": last_min_exceed_actor,

        "best_reward": best_reward,
        "last_best_reward": last_best_reward,
        "last_min_exceedance": last_min_exceedance,

        "episode_returns": episode_returns,
        "episode_exceedances": episode_exceedances,
    }