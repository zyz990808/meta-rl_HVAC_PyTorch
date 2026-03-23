__all__ = ["ContinuousBuildingControlEnvironment"]
__all__ = ["ContinuousBuildingControlEnvironment"]

import numpy as np
from scipy import signal
import pandas as pd
import gym
from gym import spaces, logger
from gym.utils import seeding

DATA_PATH = './data/'


class ContinuousBuildingControlEnvironment(gym.Env):

    def __init__(self, data_file, dt=1800, start=0., end=10000.,
                 C_env=None, C_air=None, R_rc=None, R_oe=None,
                 R_er=None, lb_set=22., ub_set=24.):

        self.dt = dt
        self.data = pd.read_csv(DATA_PATH + data_file)
        self.start = float(start)
        self.end = float(end)

        # ===== 🔥 scale initialization =====
        self.energy_scale = 1.0
        self.exceed_scale = 1.0

        # Thermal params
        self.C_env = C_env
        self.C_air = C_air
        self.R_rc = R_rc
        self.R_oe = R_oe
        self.R_er = R_er
        self.a_sol_env = 0.90303

        # State-space
        A = np.zeros((2, 2))
        B = np.zeros((2, 5))

        A[0, 0] = (-1. / self.C_env) * (1. / self.R_er + 1. / self.R_oe)
        A[0, 1] = 1. / (self.C_env * self.R_er)
        A[1, 0] = 1. / (self.C_air * self.R_er)
        A[1, 1] = (-1. / self.C_air) * (1. / self.R_er + 1. / self.R_rc)

        B[0, 1] = 1. / (self.C_env * self.R_oe)
        B[0, 2] = self.a_sol_env / self.C_env
        B[1, 0] = 1. / (self.C_air * self.R_rc)
        B[1, 2] = (1. - self.a_sol_env) / self.C_air
        B[1, 3] = 1. / self.C_air
        B[1, 4] = 1. / self.C_air

        sys = signal.StateSpace(A, B, np.array([[1, 0]]), np.zeros(5))
        discrete_matrix = sys.to_discrete(dt=self.dt)
        self.A = discrete_matrix.A
        self.B = discrete_matrix.B

        self.lb = float(lb_set)
        self.ub = float(ub_set)

        self.seed()

        self.action_space = spaces.Box(low=np.array([0.]), high=np.array([1.]), dtype=np.float32)

        self.low = np.array([10.0, 15.0, 21.0, -40.0, 0., 50., 0.])
        self.high = np.array([35.0, 28.0, 23.0, 40.0, 1100., 180., 23.])

        self.observation_space = spaces.Box(low=np.zeros((7,)), high=np.ones((7,)), dtype=np.float32)

        self.state = None
        self.t = None
        self.done = False

    # ===== 🔥 scale update =====
    def update_scale(self, energy_scale, exceed_scale):
        self.energy_scale = max(energy_scale, 1e-3)
        self.exceed_scale = max(exceed_scale, 1e-3)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, a_t):

        a = float(np.clip(a_t, 0.0, 1.0))
        s_t = self.state * (self.high - self.low) + self.low

        T_air = s_t[1]

        # ===== HVAC simplified =====
        u_t = (a - 0.5) * 3000
        energy = abs(u_t) / 1000 * 0.5

        # ===== next state =====
        s_next_room = self.A @ s_t[:2] + self.B @ np.append(s_t[2:6], u_t)
        T_air_next = float(s_next_room[-1])

        # ===== time =====
        self.t += 0.5
        idx = min(int(self.t * 2), len(self.data) - 1)

        T_out = float(self.data.iloc[idx].Tout)
        Qsg = float(self.data.iloc[idx].Qsg)
        Qint = float(self.data.iloc[idx].Qint)
        Hour = float(self.data.iloc[idx].Hour)

        s_ext = np.array([22.0, T_out, Qsg, Qint, Hour])

        # ===== comfort =====
        occupied = (7 <= Hour <= 20)

        if occupied:
            lb, ub = self.lb, self.ub
        else:
            lb, ub = 15.0, 28.0

        viol = max(lb - T_air_next, 0.0) + max(T_air_next - ub, 0.0)

        exceed = viol * 0.5

        # ===== 🔥 NEW reward =====
        E_norm = energy / self.energy_scale
        X_norm = exceed / self.exceed_scale

        reward = -(0.3 * E_norm + 0.7 * X_norm)

        # ===== state =====
        self.state = (np.concatenate([s_next_room, s_ext]) - self.low) / (self.high - self.low)

        done = self.t >= self.end
        self.done = done

        info = {
            "Energy": energy,
            "Exceedance": exceed,
            "T_room": T_air_next,
            "lb": lb,
            "ub": ub
        }

        return self.state.astype(np.float32), float(reward), done, info

    def reset(self):

        self.done = False
        self.t = self.start

        idx = min(int(self.t * 2), len(self.data) - 1)

        T_out = float(self.data.iloc[idx].Tout)
        Qsg = float(self.data.iloc[idx].Qsg)
        Qint = float(self.data.iloc[idx].Qint)
        Hour = float(self.data.iloc[idx].Hour)

        init = np.array([22.0, 22.0, 22.0, T_out, Qsg, Qint, Hour])

        self.state = (init - self.low) / (self.high - self.low)

        return self.state.astype(np.float32)