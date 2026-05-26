# Meta-RL for HVAC Control

This project studies meta-reinforcement learning (Meta-RL) for HVAC control.

The goal is to learn a generic policy that can generalize across multiple HVAC environments with different 3R2C parameters, and then quickly adapt to a new environment.

The framework mainly combines:

- PPO-based meta training
- DDPG-based inner-loop adaptation
- Warm-up mechanisms for stable adaptation
- Conservative policy updates
- Environment diversity analysis

---

# Pipeline

The workflow contains four main parts:

1. PPO-based meta-policy training
2. DDPG-based adaptation on new environments
3. Offline policy evaluation
4. Environment diversity experiments

---

# Meta-Policy Training

The meta-policy is trained using PPO across multiple HVAC environments.

Each environment has different:

- thermal capacitance
- thermal resistance
- outdoor interaction parameters

The PPO policy learns a generic control strategy that can transfer across environments.

To run meta training:

```bash
python meta-rl.py
```

The trained meta-policy will be saved in:

```text
model/
```

Main saved files include:

```text
best_actor.pth
final_actor.pth
```

---

# DDPG Adaptation

After obtaining a generic PPO policy, we use DDPG to adapt the policy to a specific environment.

The adaptation framework includes:

- random Q initialization
- warm-up training for Q learning
- conservative policy updates
- best-model selection

To run adaptation:

```bash
python ddpg_update.py
```

The adapted models will be saved in:

```text
ddpg_new_env/
```

The saved checkpoint may include:

```text
best_actor
final_actor
last_best_actor
last_min_exceed_actor
```

---

# Offline Evaluation

To evaluate the trained policy:

```bash
python offline_test.py
```

This script evaluates:

- indoor temperature control
- energy usage
- temperature exceedance
- adaptation behavior

The script also generates plots for visualization.

---

# Environment Diversity Experiments

We also study the relationship between:

- PPO training performance
- environment diversity
- convergence speed

To run diversity experiments:

```bash
python diversity.py
```

---

# Main Files

```text
meta-rl.py          PPO meta training
ddpg_update.py      DDPG adaptation
ddpg_torch.py       DDPG implementation
offline_test.py     Offline evaluation
diversity.py        Diversity experiments
Env_develop.py      HVAC environment
```

---

# Current Research Focus

Current work mainly focuses on:

- improving adaptation stability
- reducing unstable DDPG updates
- studying environment diversity
- improving meta-policy quality

---

# Repository

https://github.com/zyz990808/meta-rl_HVAC_PyTorch/tree/stability

