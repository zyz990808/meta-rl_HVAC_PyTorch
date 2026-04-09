# plot_ppo_results_cps.py
# RAW-reward compatible plotting script
# - Subplot 1 uses episode_return_total / _energy / _comfort
# - Subplot 5 decomposition uses raw TotalEnergy_kWh and TempExceed_degC
# - Reheat signal plotted as % of max (from ReheatEnergy_kWh)
# - Qsg and Qint added to Subplot 2 on a right y-axis (dual axis)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# =============================================================================
# USER SETTINGS
# =============================================================================
BASE_DIR = r"C:\Users\jbak2\OneDrive - University of Nebraska\Desktop\CPS\Connect_Env_and_basic_RL\Mar_30"
EPISODE_CSV = os.path.join(BASE_DIR, "episode_rewards.csv")
LAST_LOG_CSV = os.path.join(BASE_DIR, "last_episode_log.csv")

DT_SECONDS = 1800.0
DT_HOURS = DT_SECONDS / 3600.0

DAYS_TO_PLOT = 3
STEPS_3DAYS = int((24 * DAYS_TO_PLOT) / DT_HOURS)

# State = [T_env, T_zone, T_cor, T_out, Qsg, Qint, Hour]
OBS_LOW = np.array([10., 15., 20., -40., 0., 50., 0.], dtype=float)
OBS_HIGH = np.array([35., 28., 28.,  40., 1100., 180., 23.], dtype=float)

# Comfort box
COMFORT_LB = 21.0
COMFORT_UB = 24.0
OCC_START = 7.0
OCC_END = 20.0

# Must match env alpha (used only for decomposition label/line)
ALPHA = 0.3

# Reheat constants (must match env)
QH_REHEAT_MAX_W = 1500.0
ETA_REHEAT = 0.9
OUTER_STEP_HR = DT_HOURS

OUT_PNG = os.path.join(BASE_DIR, "ppo_results_plots.png")


# =============================================================================
# HELPERS
# =============================================================================
def denorm_obs(obs_norm: np.ndarray) -> np.ndarray:
    return obs_norm * (OBS_HIGH - OBS_LOW) + OBS_LOW


def add_occupied_boxes(ax, x_hours, y_low, y_high, occ_start=7.0, occ_end=20.0, alpha=0.12):
    max_hour = float(np.max(x_hours))
    n_days = int(np.floor(max_hour / 24.0)) + 1
    for d in range(n_days):
        x0 = d * 24.0 + occ_start
        x1 = d * 24.0 + occ_end
        if x1 < np.min(x_hours) or x0 > np.max(x_hours):
            continue
        rect = Rectangle(
            (x0, y_low),
            width=(x1 - x0),
            height=(y_high - y_low),
            facecolor="yellow",
            edgecolor=None,
            alpha=alpha,
            zorder=0,
        )
        ax.add_patch(rect)


def safe_col(df: pd.DataFrame, name: str, fallback: str = None):
    if name in df.columns:
        return df[name].to_numpy(dtype=float)
    if fallback and fallback in df.columns:
        return df[fallback].to_numpy(dtype=float)
    return None


# =============================================================================
# LOAD DATA
# =============================================================================
ep = pd.read_csv(EPISODE_CSV)
last = pd.read_csv(LAST_LOG_CSV)

episodes = ep["episode"].to_numpy(dtype=int)
ep_total = safe_col(ep, "episode_return_total", fallback="episode_return")
ep_energy = safe_col(ep, "episode_return_energy")
ep_comfort = safe_col(ep, "episode_return_comfort")

# first 3 days slice
last3 = last.iloc[:STEPS_3DAYS].copy()
tstep3 = last3["tstep"].to_numpy(dtype=int)
x_hours_3 = tstep3 * DT_HOURS

# denorm obs
obs_cols = [f"obs_{i}" for i in range(7)]
obs3 = last3[obs_cols].to_numpy(dtype=float)
obs3_den = denorm_obs(obs3)

T_zone_3 = obs3_den[:, 1]
T_out_3 = obs3_den[:, 3]
Qsg_3 = obs3_den[:, 4]
Qint_3 = obs3_den[:, 5]

# setpoints
SAT_3 = safe_col(last3, "SAT_sp", fallback="action_0")
ZAT_used_3 = safe_col(last3, "ZAT_sp_used")
if ZAT_used_3 is None:
    ZAT_used_3 = safe_col(last3, "ZAT_sp")
if ZAT_used_3 is None:
    ZAT_used_3 = safe_col(last3, "action_1")

# PI vars
m_fan_3 = safe_col(last3, "m_fan")
damper_sig_3 = safe_col(last3, "DamperSignal")

# energy (kWh per 30-min step)
E_tot_3 = safe_col(last3, "TotalEnergy_kWh")
E_cool_3 = safe_col(last3, "CoolingEnergy_kWh")
E_heat_3 = safe_col(last3, "HeatingEnergy_kWh")
E_reheat_3 = safe_col(last3, "ReheatEnergy_kWh")
E_fan_3 = safe_col(last3, "FanEnergy_kWh")

# reward decomposition (RAW)
reward_3 = safe_col(last3, "Reward", fallback="reward")
temp_exceed_3 = safe_col(last3, "TempExceed_degC")

r_energy_3 = -E_tot_3 if E_tot_3 is not None else None
r_comfort_3 = -(ALPHA * temp_exceed_3) if temp_exceed_3 is not None else None

# reheat % of max (from ReheatEnergy_kWh; convert kWh/step -> kW avg -> thermal -> %)
reheat_pct_3 = None
if E_reheat_3 is not None:
    P_reheat_avg_kW = np.array(E_reheat_3, dtype=float) / max(OUTER_STEP_HR, 1e-9)
    Q_reheat_avg_kW = P_reheat_avg_kW * ETA_REHEAT
    Qh_max_kW = QH_REHEAT_MAX_W / 1000.0
    reheat_pct_3 = 100.0 * (Q_reheat_avg_kW / max(Qh_max_kW, 1e-9))
    reheat_pct_3 = np.clip(reheat_pct_3, 0.0, 100.0)


# =============================================================================
# PLOTTING
# =============================================================================
plt.figure(figsize=(14, 18))
gs = plt.GridSpec(5, 1, height_ratios=[1.25, 2.3, 1.7, 1.7, 1.9], hspace=0.35)

# -----------------------------
# Subplot 1
# -----------------------------
ax1 = plt.subplot(gs[0])
ax1.set_title("Subplot 1: Reward over Episodes (Total + Objectives)")
if ep_total is not None:
    ax1.plot(episodes, ep_total, linewidth=1.8, label="Episode Return (Total)")
if ep_energy is not None:
    ax1.plot(episodes, ep_energy, linewidth=1.4, linestyle=":", label="Episode Return (Energy)")
if ep_comfort is not None:
    ax1.plot(episodes, ep_comfort, linewidth=1.4, linestyle="-.", label="Episode Return (Comfort)")
ax1.set_xlabel("Episode")
ax1.set_ylabel("Episode Return")
ax1.grid(True, alpha=0.3)
ax1.legend(loc="best")

# -----------------------------
# Subplot 2 (Temps/setpoints + Qsg/Qint dual y-axis)
# -----------------------------
ax2 = plt.subplot(gs[1])
ax2.set_title("Subplot 2: First 3 Days - Temps/Setpoints + Qsg/Qint (Occupied Comfort Box)")

# Left axis: temperatures + setpoints
ax2.plot(x_hours_3, T_zone_3, linewidth=2.0, label="T_zone")
ax2.plot(x_hours_3, T_out_3, linewidth=1.6, linestyle="--", color="black", label="T_out")
if SAT_3 is not None:
    ax2.plot(x_hours_3, SAT_3, linewidth=1.6, linestyle="-.", label="SAT_sp (Action)")
if ZAT_used_3 is not None:
    ax2.plot(x_hours_3, ZAT_used_3, linewidth=1.6, label="ZAT_sp (Action)", color="red")

add_occupied_boxes(ax2, x_hours_3, COMFORT_LB, COMFORT_UB, OCC_START, OCC_END, alpha=0.15)

ax2.set_xlabel("Time (hours from episode start)")
ax2.set_ylabel("Temperature / Setpoints (°C)")
ax2.grid(True, alpha=0.3)

# Right axis: Qsg/Qint
ax2b = ax2.twinx()
l_qsg, = ax2b.plot(x_hours_3, Qsg_3, linewidth=1.4, linestyle="--", label="Qsg (W)", color="green")
l_qint, = ax2b.plot(x_hours_3, Qint_3, linewidth=1.4, linestyle=":", label="Qint (W)", color="purple")
ax2b.set_ylabel("Gains (W)")

# Combined legend
lines2, labels2 = ax2.get_legend_handles_labels()
lines2b, labels2b = ax2b.get_legend_handles_labels()
ax2.legend(lines2 + lines2b, labels2 + labels2b, loc="upper right")

# -----------------------------
# Subplot 3
# -----------------------------
ax3 = plt.subplot(gs[2])
ax3.set_title("Subplot 3: First 3 Days - PI Loop Variables + Reheat (%)")
if damper_sig_3 is not None:
    ax3.plot(x_hours_3, damper_sig_3, linewidth=1.8, label="Damper Signal (%)")
ax3.set_xlabel("Time (hours)")
ax3.set_ylabel("Damper (%)")
ax3.grid(True, alpha=0.3)

ax3b = ax3.twinx()
line_handles = []
line_labels = []
if m_fan_3 is not None:
    l_mfan, = ax3b.plot(x_hours_3, m_fan_3, linewidth=1.8, linestyle="--", label="m_fan (kg/s)")
    line_handles.append(l_mfan)
    line_labels.append("m_fan (kg/s)")
ax3b.set_ylabel("m_fan (kg/s)")

ax3c = ax3.twinx()
ax3c.spines["right"].set_position(("axes", 1.10))
ax3c.spines["right"].set_visible(True)
if reheat_pct_3 is not None:
    l_reheat, = ax3c.plot(x_hours_3, reheat_pct_3, linewidth=1.8, linestyle=":", label="Reheat (% of max)")
    ax3c.set_ylim(-5, 105)
    ax3c.set_ylabel("Reheat (%)")
    line_handles.append(l_reheat)
    line_labels.append("Reheat (% of max)")

lines3, labels3 = ax3.get_legend_handles_labels()
ax3.legend(lines3 + line_handles, labels3 + line_labels, loc="upper right")

# -----------------------------
# Subplot 4
# -----------------------------
ax4 = plt.subplot(gs[3])
ax4.set_title("Subplot 4: First 3 Days - Energy per Step (kWh per 30-min)")
if E_tot_3 is not None:
    ax4.plot(x_hours_3, E_tot_3, linewidth=2.0, label="Total Energy (kWh)")
if E_cool_3 is not None:
    ax4.plot(x_hours_3, E_cool_3, linewidth=1.4, label="Cooling Energy (kWh)")
if E_heat_3 is not None:
    ax4.plot(x_hours_3, E_heat_3, linewidth=1.4, label="Heating Energy (kWh)")
if E_reheat_3 is not None:
    ax4.plot(x_hours_3, E_reheat_3, linewidth=1.4, label="Reheat Energy (kWh)")
if E_fan_3 is not None:
    ax4.plot(x_hours_3, E_fan_3, linewidth=1.4, label="Fan Energy (kWh)")
ax4.set_xlabel("Time (hours)")
ax4.set_ylabel("Energy (kWh/step)")
ax4.grid(True, alpha=0.3)
ax4.legend(loc="upper right")

# -----------------------------
# Subplot 5 (RAW decomposition)
# -----------------------------
ax5 = plt.subplot(gs[4])
ax5.set_title("Subplot 5: Reward over Timesteps (First 3 Days) + Decomposition (RAW)")
if reward_3 is not None:
    ax5.plot(x_hours_3, reward_3, linewidth=1.4, label="Reward (total) per step")
if r_energy_3 is not None:
    ax5.plot(x_hours_3, r_energy_3, linewidth=1.2, linestyle="--", label="-TotalEnergy_kWh (per step)")
if r_comfort_3 is not None:
    ax5.plot(x_hours_3, r_comfort_3, linewidth=1.2, linestyle=":", label=f"-{ALPHA}*TempExceed_degC (per step)")
ax5.set_xlabel("Time (hours)")
ax5.set_ylabel("Reward / components")
ax5.grid(True, alpha=0.3)
ax5.legend(loc="upper right")

plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
print(f"Saved figure to: {OUT_PNG}")
plt.show()