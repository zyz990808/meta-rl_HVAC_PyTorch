__all__ = ["ContinuousBuildingControlEnvironment"]

import numpy as np
from scipy import signal
import pandas as pd
import gym
from gym import spaces, logger
from gym.utils import seeding

DATA_PATH = './data/'


class ContinuousBuildingControlEnvironment(gym.Env):
    r"""
    Description:
        Continuous building control environment with indoor thermal dynamics
        and outdoor exogenous data.

    Observation (normalized to [0,1]):
        [T_env, T_air, T_cor, T_out, Qsg, Qint, Hour]

    Action:
        Scalar in [0,1] controlling HVAC input.

    Reward (per step):
        r = -Energy - Penalty
        Energy is positive electricity consumption (kWh per step).
        Penalty is positive comfort violation penalty (quadratic).
    """

    def __init__(self, data_file, dt=1800, start=0., end=10000.,
                 C_env=None, C_air=None, R_rc=None, R_oe=None,
                 R_er=None, lb_set=22., ub_set=24.):

        self.dt = dt
        self.data = pd.read_csv(DATA_PATH + data_file)
        self.start = float(start)
        self.end = float(end)
        self.num_step = int((self.end - self.start) / (self.dt / 3600) + 1)

        # Thermal parameters
        self.C_env = C_env
        self.C_air = C_air
        self.R_rc = R_rc
        self.R_oe = R_oe
        self.R_er = R_er
        self.a_sol_env = 0.90303

        # Continuous-time state-space matrices
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

        C = np.array([[1, 0]])
        D = np.zeros(5)

        sys = signal.StateSpace(A, B, C, D)
        discrete_matrix = sys.to_discrete(dt=self.dt)
        self.A = discrete_matrix.A
        self.B = discrete_matrix.B

        # Comfort band
        self.lb = float(lb_set)
        self.ub = float(ub_set)

        # HVAC parameters
        self.Tlv_cooling = 7.
        self.Tlv_heating = 35.
        self.E_cf_cooling = np.array([14.8187, -0.2538, 0.1814, -0.0003, -0.0021, 0.002])
        self.E_cf_heating = np.array([7.8885, 0.1809, -0.1568, 0.001068, 0.0009938, -0.002674])

        self.m_dot_min = 0.080938984
        self.cp_air = 1004
        self.T_sup = 16.5
        self.m_design = 0.9264 * 0.4
        self.dP = 500
        self.e_tot = 0.6045
        self.rho_air = 1.225
        self.c_FAN = np.array([0.040759894, 0.08804497, -0.07292612, 0.943739823, 0])
        self.Qh_max = 1500
        self.m_dot_max = self.m_dot_min * 550 / 140

        self.seed()

        # Action space
        self.action_space = spaces.Box(low=np.array([0.]), high=np.array([1.]), dtype=np.float32)

        # Scaling bounds for observation normalization
        self.low = np.array([10.0, 15.0, 21.0, -40.0, 0., 50., 0.], dtype=float)
        self.high = np.array([35.0, 28.0, 23.0, 40.0, 1100., 180., 23.], dtype=float)

        # Observations normalized to [0,1]
        self.observation_space = spaces.Box(low=np.zeros((7,)), high=np.ones((7,)), dtype=np.float32)

        self.state = None
        self.t = None
        self.done = False

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, a_t):
        if getattr(self, "done", False):
            logger.warn(
                "You are calling 'step()' even though this environment has already returned done = True. "
                "Call reset() after done=True."
            )
            obs = np.array(self.state) if self.state is not None else np.zeros((7,), dtype=np.float32)
            info = {
                "u_t": 0.0,
                "a_t": float(np.asarray(a_t).item()),
                "Energy": 0.0,
                "Penalty": 0.0,
                "Exceedance": 0.0,
                "T_room": float("nan"),
                "T_out": float("nan"),
                "lb": float(self.lb),
                "ub": float(self.ub),
                "viol": 0.0,
                "occupied": False,
            }
            return obs, 0.0, True, info

        # action scalar in [0,1]
        a = float(np.asarray(a_t).item())
        a = float(np.clip(a, 0.0, 1.0))

        # de-normalize current state
        s_t = self.state * (self.high - self.low) + self.low

        # COP computed using current state's outdoor temp (approx)
        T_out_for_COP = float(s_t[3])
        EFc = self.E_cf_cooling
        Tlvh = self.Tlv_heating
        Tlvc = self.Tlv_cooling

        COPh = 0.9
        COPc = (
            EFc[0]
            + T_out_for_COP * EFc[1]
            + Tlvh * EFc[2]
            + (T_out_for_COP ** 2) * EFc[3]
            + (Tlvc ** 2) * EFc[4]
            + T_out_for_COP * Tlvc * EFc[5]
        )

        # HVAC thermal input and base energy (kWh/step)
        if abs(a - 0.5) < 1e-12:
            u_t = self.m_dot_min * self.cp_air * (self.T_sup - s_t[1])
            m_fan = self.m_dot_min
            energy = (-1.0 * u_t / COPc) / 1000.0 / 2.0
        elif a > 0.5:
            u_h = ((a - 0.5) / 0.5) * self.Qh_max
            u_t = u_h + self.m_dot_min * self.cp_air * (self.T_sup - s_t[1])
            m_fan = self.m_dot_min
            energy = (u_h / COPh + self.m_dot_min * self.cp_air * (s_t[1] - self.T_sup) / COPc) / 1000.0 / 2.0
        else:
            m_fan = (self.m_dot_max - self.m_dot_min) * ((0.5 - a) / 0.5) + self.m_dot_min
            u_t = m_fan * self.cp_air * (self.T_sup - s_t[1])
            energy = (-1.0 * u_t / COPc) / 1000.0 / 2.0

        # next room states
        s_next_room = self.A @ s_t[:2] + self.B @ np.append(s_t[2:6], u_t)
        T_air_next = float(s_next_room[-1])

        # fan power add-on (kWh/step)
        f_flow = m_fan / self.m_design
        f_pl = (
            self.c_FAN[0]
            + self.c_FAN[1] * f_flow
            + self.c_FAN[2] * f_flow ** 2
            + self.c_FAN[3] * f_flow ** 3
            + self.c_FAN[4] * f_flow ** 4
        )
        Q_fan = f_pl * self.m_design * self.dP / (self.e_tot * self.rho_air)
        energy += Q_fan / 1000.0 / 2.0

        # force positive electricity consumption
        energy = float(abs(energy))

        # advance time (0.5 hour per step)
        self.t = float(self.t + 0.5)

        # safe index into data (2 samples per hour)
        idx = int(self.t * 2)
        if idx < 0:
            idx = 0
        if idx >= len(self.data):
            idx = len(self.data) - 1

        # exogenous states from data at new time
        T_cor = 22.0
        T_out = float(self.data.iloc[idx].Tout)
        Qsg = float(self.data.iloc[idx].Qsg)
        Qint = float(self.data.iloc[idx].Qint)
        Hour = float(self.data.iloc[idx].Hour)
        s_ext = np.array([T_cor, T_out, Qsg, Qint, Hour], dtype=float)

        # comfort logic
        occupied = (7 <= Hour <= 20)
        if occupied:
            lb = float(self.lb)
            ub = float(self.ub)
            temp_weight = 5.0   # tune
        else:
            lb = 15.0
            ub = 28.0
            temp_weight = 0.0

        viol_low = max(lb - T_air_next, 0.0)
        viol_high = max(T_air_next - ub, 0.0)
        viol = float(viol_low + viol_high)

        Temp_exceed = float(viol * 0.5)          # 0.5 hr per step
        temp_penalty = float(temp_weight * (viol ** 2))

        # reward: always discourage energy and discomfort
        r = -energy - temp_penalty

        # normalized next state
        self.state = (np.concatenate([s_next_room, s_ext]) - self.low) / (self.high - self.low)

        done = bool(self.t + 0.5 >= self.end)
        self.done = done

        info = {
            "u_t": float(u_t),
            "a_t": float(a),
            "Energy": float(energy),
            "Penalty": float(temp_penalty),
            "Exceedance": float(Temp_exceed),
            "T_room": float(T_air_next),
            "T_out": float(T_out),
            "lb": float(lb),
            "ub": float(ub),
            "viol": float(viol),
            "occupied": bool(occupied),
        }

        return np.array(self.state, dtype=np.float32), float(r), done, info

    def reset(self):
        self.done = False
        self.t = float(self.start)

        # initial condition
        T_env_0 = 22.0
        T_air_0 = 22.0
        T_cor = 22.0

        idx = int(self.t * 2)
        if idx < 0:
            idx = 0
        if idx >= len(self.data):
            idx = len(self.data) - 1

        T_out = float(self.data.iloc[idx].Tout)
        Qsg = float(self.data.iloc[idx].Qsg)
        Qint = float(self.data.iloc[idx].Qint)
        Hour = float(self.data.iloc[idx].Hour)

        self.state = (np.array([T_env_0, T_air_0, T_cor, T_out, Qsg, Qint, Hour], dtype=float) - self.low) / (self.high - self.low)
        return np.array(self.state, dtype=np.float32)