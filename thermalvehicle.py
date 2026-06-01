"""
整车车速跟踪 -> 双电机(效率/发热/两状态热) -> 电池(你给的 BatteryPackModel)

耦合逻辑（你要求的）：
1) 车速参考输入给“动力学+电机”模型，得到前/后电机扭矩、转速、效率
2) 由电机机械功率与效率得到电机电功率 P_elec_f/P_elec_r
3) 总电功率 P_elec_total / 电池端电压 V_batt -> 电池电流 I_batt，作为 BatteryPackModel 的输入
4) 冷却液入口温度（前电机/后电机/电池）先都用常数（可独立设置）
5) 输出并绘图：你列的所有曲线

注意：
- 你的 BatteryPackModel 约定：I>0 => SOC 增加（和常见放电为正相反）
  因此这里我们把“驱动消耗电能”定义为 I_batt < 0（SOC 会下降）。
- 电机热模型：每台电机有 Tm(电机金属温度) 与 Tc(内部1L冷却液温度，且 Tout=Tc)
  入口温度 Tin 为常数输入，出口温度 Tout 由模型计算。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator, interp1d


# =========================================================
# 工具函数
# =========================================================
def sat(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)


def load_speed_profile_time_v(csv_path, t_col=0, v_col=1, has_header=True, encoding=None, sep=","):
    df = pd.read_csv(csv_path, encoding=encoding, sep=sep, header=0 if has_header else None)
    t_s = df.iloc[:, t_col].to_numpy(dtype=float)
    v_kmh = df.iloc[:, v_col].to_numpy(dtype=float)
    idx = np.argsort(t_s)
    return t_s[idx], v_kmh[idx]


class SpeedProfile:
    def __init__(self, t_s, v_kmh):
        self.t = np.asarray(t_s, dtype=float)
        self.v = np.asarray(v_kmh, dtype=float)

    def v_ref(self, t_now_s):
        return float(np.interp(t_now_s, self.t, self.v, left=self.v[0], right=self.v[-1]))

    @property
    def t_end(self):
        return float(self.t[-1])


def load_eff_map_csv_A1_empty_Acol_torque_Brow_speed(csv_path, fill_eta=0.2, encoding=None, sep=","):
    df = pd.read_csv(csv_path, header=None, encoding=encoding, sep=sep)
    speed_rpm = df.iloc[0, 1:].to_numpy(dtype=float)
    torque_nm = df.iloc[1:, 0].to_numpy(dtype=float)
    eta = df.iloc[1:, 1:].to_numpy(dtype=float)
    eta = np.where(np.isnan(eta), float(fill_eta), eta)
    if np.nanmax(eta) > 1.5:
        eta = eta / 100.0
    eta = sat(eta, 0.0, 1.0)

    if np.any(np.diff(torque_nm) < 0):
        idx = np.argsort(torque_nm)
        torque_nm = torque_nm[idx]
        eta = eta[idx, :]
    if np.any(np.diff(speed_rpm) < 0):
        jdx = np.argsort(speed_rpm)
        speed_rpm = speed_rpm[jdx]
        eta = eta[:, jdx]

    itp = RegularGridInterpolator((torque_nm, speed_rpm), eta, bounds_error=False, fill_value=None)
    return torque_nm, speed_rpm, eta, itp


def motor_loss_power_from_eff(Pmech_W, eta):
    eta = float(np.clip(eta, 1e-4, 1.0))
    Pmech_W = float(Pmech_W)
    if Pmech_W >= 0:
        return Pmech_W * (1.0 / eta - 1.0)
    else:
        return (-Pmech_W) * (1.0 - eta)


def motor_elec_power_from_eff(Pmech_W, eta):
    """
    返回电功率 Pelec（W），符号约定：
    - 驱动：Pmech>=0 => Pelec>0（从电池取电）
    - 回收：Pmech<0  => Pelec<0（回充电池）
    """
    eta = float(np.clip(eta, 1e-4, 1.0))
    Pmech_W = float(Pmech_W)
    if Pmech_W >= 0:
        return Pmech_W / eta
    else:
        return Pmech_W * eta


def invR(R):
    return 0.0 if (not np.isfinite(R) or R <= 0) else 1.0 / R


# =========================================================
# 电机 map 集合
# =========================================================
class MotorMapSet:
    def __init__(self, drive_map, brake_map):
        self.Td_axis, self.nd_axis, _, self.itp_d = drive_map
        self.Tb_axis, self.nb_axis, _, self.itp_b = brake_map

        self.n_min = float(min(np.min(self.nd_axis), np.min(self.nb_axis)))
        self.n_max = float(max(np.max(self.nd_axis), np.max(self.nb_axis)))

        self.T_drive_max = float(np.max(self.Td_axis))
        self.T_brake_min = float(np.min(self.Tb_axis))

        self._Td_min, self._Td_max = float(np.min(self.Td_axis)), float(np.max(self.Td_axis))
        self._Tb_min, self._Tb_max = float(np.min(self.Tb_axis)), float(np.max(self.Tb_axis))
        self._nd_min, self._nd_max = float(np.min(self.nd_axis)), float(np.max(self.nd_axis))
        self._nb_min, self._nb_max = float(np.min(self.nb_axis)), float(np.max(self.nb_axis))

    def clamp_speed(self, n_rpm):
        return float(sat(n_rpm, self.n_min, self.n_max))

    def clamp_torque(self, T_nm):
        return float(sat(T_nm, self.T_brake_min, self.T_drive_max))

    def eta(self, T_nm, n_rpm):
        if T_nm >= 0:
            Tq = float(sat(T_nm, self._Td_min, self._Td_max))
            nq = float(sat(n_rpm, self._nd_min, self._nd_max))
            return float(self.itp_d([[Tq, nq]])[0])
        else:
            Tq = float(sat(T_nm, self._Tb_min, self._Tb_max))
            nq = float(sat(n_rpm, self._nb_min, self._nb_max))
            return float(self.itp_b([[Tq, nq]])[0])


# =========================================================
# 双电机 + 纵向动力学（车速跟踪）
# =========================================================
class DiscreteDualMotorLongitudinalModel:
    def __init__(
        self,
        dt,
        m, Rw,
        Crr=0.0, rho=1.225, A=0.0, Cd=0.0,
        theta=0.0, delta=0.0,
        i_f=9.9, i_r=11.0,
        front_maps=None, rear_maps=None,
        v0_kmh=0.0,
    ):
        if front_maps is None or rear_maps is None:
            raise ValueError("front_maps and rear_maps must be provided.")
        self.dt = float(dt)
        self.m = float(m)
        self.Rw = float(Rw)
        self.Crr = float(Crr)
        self.rho = float(rho)
        self.A = float(A)
        self.Cd = float(Cd)
        self.theta = float(theta)
        self.delta = float(delta)
        self.i_f = float(i_f)
        self.i_r = float(i_r)
        self.front = front_maps
        self.rear = rear_maps
        self.g = 9.81
        self.v = float(v0_kmh) / 3.6

    def reset(self, v0_kmh=0.0):
        self.v = float(v0_kmh) / 3.6

    def _resist_forces(self, v, theta):
        F_roll = self.m * self.g * self.Crr * np.cos(theta)
        F_aero = 0.5 * self.rho * self.A * self.Cd * v * v
        F_grade = self.m * self.g * np.sin(theta)
        return F_roll + F_aero + F_grade

    def _motor_speeds_rpm(self, v):
        w_w = v / self.Rw if self.Rw > 0 else 0.0
        n_f = (self.i_f * w_w) * 60.0 / (2.0 * np.pi)
        n_r = (self.i_r * w_w) * 60.0 / (2.0 * np.pi)
        return self.front.clamp_speed(n_f), self.rear.clamp_speed(n_r)

    @staticmethod
    def _split_by_front_share(Tw_total, alpha_f):
        alpha_f = float(sat(float(alpha_f), 0.0, 1.0))
        Tw_f = alpha_f * Tw_total
        Tw_r = (1.0 - alpha_f) * Tw_total
        return Tw_f, Tw_r, alpha_f

    def step_with_speed_ref(self, v_ref_kmh, alpha_f, Kp=800.0, theta=None):
        theta = self.theta if theta is None else float(theta)
        v_ref = float(v_ref_kmh) / 3.6
        e = v_ref - self.v
        Tw_total_cmd = Kp * e

        v0 = self.v
        n_mf, n_mr = self._motor_speeds_rpm(v0)

        Tw_f_cmd, Tw_r_cmd, alpha_f_used = self._split_by_front_share(Tw_total_cmd, alpha_f)

        T_mf_req = Tw_f_cmd / self.i_f
        T_mr_req = Tw_r_cmd / self.i_r

        T_mf = self.front.clamp_torque(T_mf_req)
        T_mr = self.rear.clamp_torque(T_mr_req)

        eta_f = self.front.eta(T_mf, n_mf)
        eta_r = self.rear.eta(T_mr, n_mr)

        w_mf = n_mf * 2.0 * np.pi / 60.0
        w_mr = n_mr * 2.0 * np.pi / 60.0
        Pmech_f = T_mf * w_mf
        Pmech_r = T_mr * w_mr

        Ploss_f = motor_loss_power_from_eff(Pmech_f, eta_f)
        Ploss_r = motor_loss_power_from_eff(Pmech_r, eta_r)
        Pelec_f = motor_elec_power_from_eff(Pmech_f, eta_f)
        Pelec_r = motor_elec_power_from_eff(Pmech_r, eta_r)

        Tw_act = T_mf * self.i_f + T_mr * self.i_r
        Fx_drive = Tw_act / self.Rw if self.Rw > 0 else 0.0

        Fres = self._resist_forces(v0, theta)
        a = (Fx_drive - Fres) / (self.m * (1.0 + self.delta))
        v1 = max(0.0, v0 + a * self.dt)
        self.v = v1

        return {
            "v_kmh": v1 * 3.6,
            "n_mf_rpm": n_mf,
            "n_mr_rpm": n_mr,
            "T_mf_Nm": float(T_mf),
            "T_mr_Nm": float(T_mr),
            "eta_f": float(eta_f),
            "eta_r": float(eta_r),
            "Pmech_f_W": float(Pmech_f),
            "Pmech_r_W": float(Pmech_r),
            "Pelec_f_W": float(Pelec_f),
            "Pelec_r_W": float(Pelec_r),
            "Ploss_f_W": float(Ploss_f),
            "Ploss_r_W": float(Ploss_r),
            "Tw_total_cmd_Nm": float(Tw_total_cmd),
            "Tw_total_act_Nm": float(Tw_act),
            "alpha_f_used": float(alpha_f_used),
        }


# =========================================================
# 电机两状态热模型：Tm(电机) + Tc(内部1L冷却液，Tout=Tc)
# =========================================================
class MotorCoolantTwoStateThermal:
    def __init__(
        self,
        dt,
        m_motor_kg=300.0,
        c_motor_J_per_kgK=450.0,
        V_coolant_L=1.0,
        rho_coolant_kg_per_m3=1060.0,
        cp_coolant_J_per_kgK=3600.0,
        R_mc_K_per_W=0.05,
        R_ma_K_per_W=np.inf,
        Tm0_degC=25.0,
        Tc0_degC=25.0,
        Tmin_degC=-40.0,
        Tmax_degC=250.0,
    ):
        self.dt = float(dt)
        self.Cm = float(m_motor_kg) * float(c_motor_J_per_kgK)
        V_m3 = float(V_coolant_L) * 1e-3
        m_cool = float(rho_coolant_kg_per_m3) * V_m3
        self.Cc = m_cool * float(cp_coolant_J_per_kgK)
        self.cp = float(cp_coolant_J_per_kgK)
        self.R_mc = float(R_mc_K_per_W)
        self.R_ma = float(R_ma_K_per_W)
        self.Tm = float(Tm0_degC)
        self.Tc = float(Tc0_degC)
        self.Tmin = float(Tmin_degC)
        self.Tmax = float(Tmax_degC)

    @property
    def UA_mc(self):
        return invR(self.R_mc)

    @property
    def UA_ma(self):
        return invR(self.R_ma)

    def reset(self, Tm0_degC=25.0, Tc0_degC=25.0):
        self.Tm = float(Tm0_degC)
        self.Tc = float(Tc0_degC)

    def step(self, Ploss_W, mdot_kgps, Tin_degC, Tamb_degC):
        Ploss = float(Ploss_W)
        mdot = max(0.0, float(mdot_kgps))
        Tin = float(Tin_degC)
        Tamb = float(Tamb_degC)
        Tm = self.Tm
        Tc = self.Tc

        Q_mc = self.UA_mc * (Tm - Tc)
        Q_ma = self.UA_ma * (Tm - Tamb)
        Q_flow = mdot * self.cp * (Tin - Tc)

        dTm = (Ploss - Q_mc - Q_ma) / self.Cm
        dTc = (Q_mc + Q_flow) / self.Cc

        self.Tm = float(np.clip(Tm + self.dt * dTm, self.Tmin, self.Tmax))
        self.Tc = float(np.clip(Tc + self.dt * dTc, self.Tmin, self.Tmax))

        return {"Tm_degC": self.Tm, "Tin_degC": Tin, "Tout_degC": self.Tc}


# =========================================================
# 你的 BatteryPackModel（保持你原始接口：step(I_A, Tin_degC, mdot_kgps)）
# 说明：这里只把 reset 修正为真正写回 self.x（其它不动）
# =========================================================
class BatteryPackModel:
    def __init__(
        self,
        dt=1.0,
        Ns=102*2,
        Q_Ah=19.5*5,
        SOC0=0.6, Tb0_degC=40.0, Up0_V=0.0, Tc0_degC=40.0,
        mass_kg=None,
        c_batt_J_per_kgK=1100.0,
        Tamb_degC=40.0,
        Rc_K_per_W=0.05,
        h_W_per_m2K=1.0,
        S_m2=3,
        Vc_m3=0.002,
        rho_cool_kg_per_m3=1050.0,
        cp_cool_J_per_kgK=3400.0,
        UA_cool_to_amb_W_per_K=0.27 * 50,
        soc_clip=(0.0, 1.0),
    ):
        self.dt = float(dt)
        self.Ns = int(Ns)
        self.Q_Ah = float(Q_Ah)
        self.mass_kg = float(0.547 * Ns if mass_kg is None else mass_kg)
        self.c_batt = float(c_batt_J_per_kgK)
        self.Tamb = float(Tamb_degC)
        self.Rc = float(Rc_K_per_W)
        self.h = float(h_W_per_m2K)
        self.S = float(S_m2)
        self.Vc = float(Vc_m3)
        self.rho_cool = float(rho_cool_kg_per_m3)
        self.cp_cool = float(cp_cool_J_per_kgK)
        self.UA_cool_amb = float(UA_cool_to_amb_W_per_K)
        self.soc_lo, self.soc_hi = soc_clip

        SOC_grid_11 = np.arange(0.0, 1.01, 0.1)

        dOCVdTdata = np.array([
            -0.000646183, -0.000220223, -0.000143354, -7.93379e-05, 0.000101182,
            0.000149733, 0.000153284, 3.1834e-05, 1.11744e-05, 2.34698e-05, 4.85897e-05
        ], dtype=float)
        self._dOCVdT_1d = interp1d(SOC_grid_11, dOCVdTdata, kind="linear", fill_value="extrapolate")

        OCVdata = np.array([3.23, 3.43065, 3.508497, 3.57854, 3.622303,
                            3.66421, 3.735878, 3.855272, 3.957466, 4.069646, 4.193695], dtype=float)
        self._OCV_1d = interp1d(SOC_grid_11, OCVdata, kind="linear", fill_value="extrapolate")

        Tb_grid = np.array([-20, 25, 55], dtype=float)
        I_grid = np.array([9.75, 19.5, 55], dtype=float)

        Romdata = np.array([
            0.001748, 0.001898, 0.002005, 0.001448, 0.001598, 0.001705, 0.001245, 0.001345,
            0.001446, 0.001163, 0.001249, 0.001324, 0.001137, 0.001209, 0.001267, 0.001128,
            0.001207, 0.001273, 0.001119, 0.001253, 0.001373, 0.001108, 0.001258, 0.001407,
            0.001106, 0.001237, 0.001351, 0.001111, 0.001234, 0.001345, 0.001132, 0.001264,
            0.001383, 0.000893, 0.000874, 0.000846, 0.000793, 0.000774, 0.000746, 0.000697,
            0.000679, 0.000663, 0.000663, 0.00066, 0.000658, 0.000654, 0.000647, 0.000642,
            0.00065, 0.00064, 0.000636, 0.000645, 0.000644, 0.000645, 0.000639, 0.000635,
            0.00063, 0.000635, 0.00063, 0.000625, 0.000632, 0.000632, 0.000628, 0.000634,
            0.000631, 0.00063, 0.000954, 0.001053, 0.001109, 0.000754, 0.000853, 0.000909,
            0.000717, 0.0008, 0.000856, 0.000694, 0.000773, 0.00084, 0.000696, 0.000751,
            0.000799, 0.000695, 0.000763, 0.000814, 0.000696, 0.000813, 0.000913, 0.000703,
            0.00082, 0.000937, 0.000695, 0.000799, 0.000896, 0.000694, 0.000801, 0.000902,
            0.000705, 0.000817, 0.000925
        ], dtype=float).reshape(3, 11, 3)
        self._Rom = RegularGridInterpolator((Tb_grid, SOC_grid_11, I_grid), Romdata,
                                            bounds_error=False, fill_value=float(np.max(Romdata)))

        SOC_grid_10 = np.arange(0.1, 1.01, 0.1)

        Rpdata = np.array([
            0.001221, 0.001153, 0.001064, 0.000902, 0.000893, 0.000896,
            0.000775, 0.000774, 0.000787, 0.000716, 0.000728, 0.000721,
            0.000807, 0.000751, 0.000717, 0.001042, 0.001009, 0.000911,
            0.0012, 0.001321, 0.001294, 0.00114, 0.001206, 0.001192,
            0.001078, 0.001123, 0.001107, 0.001054, 0.001033, 0.001024,
            0.000961, 0.000955, 0.000938, 0.000695, 0.000674, 0.000679,
            0.000585, 0.000585, 0.000607, 0.000555, 0.000537, 0.000537,
            0.000567, 0.000559, 0.000543, 0.000758, 0.000769, 0.000736,
            0.00091, 0.000999, 0.000998, 0.000819, 0.000863, 0.000872,
            0.000768, 0.000826, 0.000811, 0.000754, 0.000789, 0.000779,
            0.000705, 0.000494, 0.00043, 0.000574, 0.000472, 0.000428,
            0.000473, 0.000495, 0.000484, 0.000455, 0.000447, 0.000444,
            0.000491, 0.000456, 0.000432, 0.000695, 0.000633, 0.000562,
            0.000783, 0.000852, 0.000861, 0.00069, 0.000699, 0.000733,
            0.000665, 0.000662, 0.000678, 0.000695, 0.000655, 0.000664
        ], dtype=float).reshape(3, 10, 3)
        self._Rp = RegularGridInterpolator((Tb_grid, SOC_grid_10, I_grid), Rpdata,
                                           bounds_error=False, fill_value=float(np.max(Rpdata)))

        Cpdata = np.array([
            7271.648, 6975.292, 6641.619, 10045.53, 9807.946, 9954.464, 11231.4, 11412.54, 11442.25,
            11376.62, 11690.69, 11566.74, 11552.67, 11164.92, 11090.48, 9678.515, 9865.793, 9957.203,
            8849.135, 9072.436, 8924.505, 9085.144, 9385.937, 9132.831, 9258.773, 9495.711, 9382.932,
            9606.074, 9681.239, 9773.668, 9656.957, 9540.146, 9279.953, 12861.1, 12433.97, 12481.51,
            15115.31, 14992.29, 15348.09, 16167.47, 15669.57, 15557.61, 15186.31, 14791.13, 14625.63,
            12568.99, 12922.25, 13035.13, 11375.08, 11953.83, 11872.24, 11688.48, 12056.36, 12060.47,
            11967.36, 12531.86, 12370.67, 12658.02, 12986.7, 13006.54, 12905.26, 11764.3, 13040.05,
            15824.3, 15790.08, 16003.37, 17046.58, 18291.56, 17973.91, 18286.21, 18094.78, 18259.38,
            17096.35, 17167.16, 16935.99, 14369.01, 14386.51, 14450.51, 13532.81, 13692.44, 13236.42,
            13554.72, 13592.69, 13726.05, 13817.37, 13963.12, 14169.65, 14745.54, 14367.23, 14717.39
        ], dtype=float).reshape(3, 10, 3)
        self._Cp = RegularGridInterpolator((Tb_grid, SOC_grid_10, I_grid), Cpdata,
                                           bounds_error=False, fill_value=float(np.max(Cpdata)))

        self.x = np.array([SOC0, Tb0_degC, Up0_V, Tc0_degC], dtype=float)

    def reset(self, SOC0=0.6, Tb0_degC=40.0, Up0_V=0.0, Tc0_degC=40.0):
        self.x[:] = [float(SOC0), float(Tb0_degC), float(Up0_V), float(Tc0_degC)]

    def step(self, I_A, Tin_degC, mdot_kgps):
        SOC0, Tb0, Up0, Tc0 = self.x
        dt = self.dt
        I = float(I_A)
        Tin = float(Tin_degC)
        mdot = max(0.0, float(mdot_kgps))

        SOC = SOC0 + I * dt / (3600.0 * self.Q_Ah)
        SOC = float(np.clip(SOC, self.soc_lo, self.soc_hi))

        OCV = self.Ns * float(self._OCV_1d(SOC))
        dOCVdT = self.Ns * float(self._dOCVdT_1d(SOC))

        Tb = Tb0
        Iabs = abs(I)

        Rom = float(self._Rom([[Tb, SOC, Iabs]])[0]) * self.Ns
        SOC_q = float(np.clip(SOC, 0.1, 1.0))
        Rp = float(self._Rp([[Tb, SOC_q, Iabs]])[0]) * self.Ns
        Cp = float(self._Cp([[Tb, SOC_q, Iabs]])[0]) / self.Ns

        Up = Up0 + (I / Cp - Up0 / (Rp * Cp)) * dt
        Vt = OCV + Up + I * Rom

        heat_W = 1.2 * abs(I * (OCV - Vt)) - I * Tb0 * dOCVdT
        heat_W *= 1.1

        disheat_W = self.h * self.S * (Tb0 - self.Tamb) + (Tb0 - Tc0) / self.Rc
        Tb = Tb0 + (heat_W - disheat_W) * dt / (self.mass_kg * self.c_batt)

        denom = self.Vc * self.rho_cool * self.cp_cool
        Tc = Tc0 + (
            mdot * self.cp_cool * (Tin - Tc0)
            + (Tb - Tc0) / self.Rc
            + self.UA_cool_amb * (self.Tamb - Tc0)
        ) * dt / denom

        self.x[:] = [SOC, Tb, Up, Tc]

        return {
            "Vt_V": float(Vt),
            "SOC": float(SOC),
            "Tb_degC": float(Tb),
            "Tout_degC": float(Tc),
            "Up_V": float(Up),
            "heat_W": float(heat_W),
        }

    def peek_voltage(self, I_A=0.0):
        """
        在不推进状态的情况下，基于当前状态估算端电压 Vt。
        用于 I≈P/V 的 V 估算，避免用 step() 产生状态扰动。
        """
        SOC0, Tb0, Up0, Tc0 = self.x
        I = float(I_A)

        SOC = float(np.clip(SOC0, self.soc_lo, self.soc_hi))

        OCV = self.Ns * float(self._OCV_1d(SOC))

        Tb = Tb0
        Iabs = abs(I)

        Rom = float(self._Rom([[Tb, SOC, Iabs]])[0]) * self.Ns
        SOC_q = float(np.clip(SOC, 0.1, 1.0))
        Rp = float(self._Rp([[Tb, SOC_q, Iabs]])[0]) * self.Ns
        Cp = float(self._Cp([[Tb, SOC_q, Iabs]])[0]) / self.Ns

        # 极化电压状态使用当前 Up0（不更新）
        Up = float(Up0)

        Vt = OCV + Up + I * Rom
        return float(Vt)



def load_cop_map_excel_A1_empty_Tamb_col_Qrow(
            xlsx_path,
            sheet_name=0,
            fill_cop=1.0,
            q_row=1,  # <-- 你的Q轴在第1行
            tamb_start_row=2,  # <-- 你的Tamb从第2行开始
    ):
        df = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=None)

        # 读轴与表（用你文件的实际位置）
        Q_axis = pd.to_numeric(df.iloc[q_row, 1:], errors="coerce").to_numpy(dtype=float)  # 第1行B..：Q
        Tamb_axis = pd.to_numeric(df.iloc[tamb_start_row:, 0], errors="coerce").to_numpy(dtype=float)  # 第2行A..：Tamb
        COP = df.iloc[tamb_start_row:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

        # 删除 NaN 列/行（防止末尾空白）
        row_valid = ~np.isnan(Tamb_axis)
        col_valid = ~np.isnan(Q_axis)
        Tamb_axis = Tamb_axis[row_valid]
        Q_axis = Q_axis[col_valid]
        COP = COP[row_valid, :][:, col_valid]

        if Tamb_axis.size == 0 or Q_axis.size == 0:
            raise ValueError(
                f"COP map axis empty. Tamb_axis.size={Tamb_axis.size}, Q_axis.size={Q_axis.size}."
            )

        COP = np.where(np.isnan(COP), float(fill_cop), COP)
        COP = np.clip(COP, 0.1, 50.0)

        # 去重+排序，保证严格递增
        Tamb_unique, idx = np.unique(Tamb_axis, return_index=True)
        idx = np.sort(idx)
        Tamb_axis = Tamb_axis[idx]
        COP = COP[idx, :]

        Q_unique, jdx = np.unique(Q_axis, return_index=True)
        jdx = np.sort(jdx)
        Q_axis = Q_axis[jdx]
        COP = COP[:, jdx]

        idx = np.argsort(Tamb_axis)
        Tamb_axis = Tamb_axis[idx]
        COP = COP[idx, :]

        jdx = np.argsort(Q_axis)
        Q_axis = Q_axis[jdx]
        COP = COP[:, jdx]

        if np.any(np.diff(Tamb_axis) <= 0) or np.any(np.diff(Q_axis) <= 0):
            raise ValueError(f"Axis not strictly increasing. Tamb_axis={Tamb_axis}, Q_axis={Q_axis}")

        itp = RegularGridInterpolator((Tamb_axis, Q_axis), COP, bounds_error=False, fill_value=None)
        return Tamb_axis, Q_axis, COP, itp
'''
def heatpump_ptc_step_Qcmd(
    T_return_degC,
    mdot_kgps,
    Tamb_degC,
    Q_hp_cmd_W,       # 热泵给水的制热量指令（W）
    P_ptc_elec_W,     # PTC电功率（W），COP=1 => Q_ptc = P_ptc
    cop_itp,          # 上面读出来的 RegularGridInterpolator
    cp_cool_J_per_kgK=3600.0,
):
    """
    输入：
      T_return_degC : 回水温度（混合后进入热泵的水温）
      mdot_kgps     : 主回路质量流量（kg/s）
      Tamb_degC     : 环境温度（degC）
      Q_hp_cmd_W    : 热泵制热量（给水增加的热量，W）
      P_ptc_elec_W  : PTC电功率（W）
      cop_itp       : COP(Tamb, Qheat) 插值器

    输出：
      T_batt_in_degC : 经过热泵+PTC后的水温（进入电池）
      P_hp_elec_W    : 热泵电功率消耗（W）
      P_ptc_elec_W   : PTC电功率消耗（W）
      COP            : 当前COP
    """
    mdot = max(0.0, float(mdot_kgps))
    Tret = float(T_return_degC)
    Tamb = float(Tamb_degC)

    Q_hp = max(0.0, float(Q_hp_cmd_W))
    P_ptc = max(0.0, float(P_ptc_elec_W))
    Q_ptc = P_ptc  # COP=1

    # 无流量时：不改变水温；能耗仍可记录（也可改成强制0）
    if mdot <= 1e-9:
        return {
            "T_batt_in_degC": Tret,
            "P_hp_elec_W": 0.0 if Q_hp > 0 else 0.0,
            "P_ptc_elec_W": P_ptc,
            "Q_hp_to_water_W": 0.0,
            "Q_ptc_to_water_W": 0.0,
            "COP": np.nan,
        }

    # 查COP：COP = f(Tamb, Q_hp)
    COP = float(cop_itp([[Tamb, Q_hp]])[0])
    COP = float(np.clip(COP, 0.1, 50.0))

    # 热泵电功率：P = Q/COP
    P_hp = Q_hp / COP

    # 水温升高
    dT = (Q_hp + Q_ptc) / (mdot * cp_cool_J_per_kgK)
    T_batt_in = Tret + dT

    return {
        "T_batt_in_degC": T_batt_in,
        "P_hp_elec_W": P_hp,
        "P_ptc_elec_W": P_ptc,
        "Q_hp_to_water_W": Q_hp,
        "Q_ptc_to_water_W": Q_ptc,
        "COP": COP,
    }
'''
class HeatPumpPTC2L:
    """
    热泵+PTC + 内部2L固定液体（完全混合）热模型（离散时间）
    状态：T_hp (degC) = 模块内2L液体温度 = 供水温度 = 电池入口温度

    输入：
      T_return_degC, mdot_kgps, Tamb_degC, Q_hp_cmd_W, P_ptc_elec_W, cop_itp

    输出：
      T_batt_in_degC (=T_hp), P_hp_elec_W, P_ptc_elec_W, COP, Q_hp_to_water_W, Q_ptc_to_water_W
    """

    def __init__(self, dt,
                 V_L=2.0,
                 rho_kg_per_m3=1060.0,
                 cp_J_per_kgK=3600.0,
                 T0_degC=25.0):
        self.dt = float(dt)
        self.cp = float(cp_J_per_kgK)
        V_m3 = float(V_L) * 1e-3
        m = float(rho_kg_per_m3) * V_m3
        self.C = m * self.cp               # J/K
        self.T = float(T0_degC)            # 状态温度

    def reset(self, T0_degC=25.0):
        self.T = float(T0_degC)

    def step(self, T_return_degC, mdot_kgps, Tamb_degC,
             Q_hp_cmd_W, P_ptc_elec_W, cop_itp):
        mdot = max(0.0, float(mdot_kgps))
        Tret = float(T_return_degC)
        Tamb = float(Tamb_degC)

        Q_hp = max(0.0, float(Q_hp_cmd_W))
        P_ptc = max(0.0, float(P_ptc_elec_W))
        Q_ptc = P_ptc  # COP=1

        # ---- COP查表（建议夹到map范围，防止外推异常）----
        # RegularGridInterpolator 在 scipy 里可通过 .grid 取轴
        Tamb_axis, Q_axis = cop_itp.grid
        Tamb_q = float(np.clip(Tamb, Tamb_axis[0], Tamb_axis[-1]))
        Q_q = float(np.clip(Q_hp, Q_axis[0], Q_axis[-1]))

        COP = float(cop_itp([[Tamb_q, Q_q]])[0])
        COP = float(np.clip(COP, 0.1, 50.0))

        # 热泵电功率
        P_hp = (Q_hp / COP) if Q_hp > 0 else 0.0

        # ---- 2L控制体能量平衡（完全混合）----
        # C*dT = mdot*cp*(Tret - T) + Q_hp + Q_ptc
        Q_flow = mdot * self.cp * (Tret - self.T)
        dT = (Q_flow + Q_hp + Q_ptc) * (self.dt / self.C)
        self.T = self.T + dT

        return {
            "T_batt_in_degC": self.T,
            "P_hp_elec_W": P_hp,
            "P_ptc_elec_W": P_ptc,
            "Q_hp_to_water_W": Q_hp,
            "Q_ptc_to_water_W": Q_ptc,
            "COP": COP,
            "Tret_degC": Tret,
        }
def coolant_network_step(
    mode: int,
    mdot_total_kgps: float,
    gamma_split: float,
    # temperatures available at current step k (from previous updates or initial states)
    Tout_batt_degC: float,
    Tout_f_degC: float,
    Tout_r_degC: float,
):
    """
    根据模式，计算：
      - 电机入口温度 Tin_f/Tin_r
      - 电机流量 mdot_f/mdot_r
      - 回水混合温度 T_return (送热泵+PTC)
      - 电池流量 mdot_batt (=mdot_total)

    注意：这里把“回水混合点”的组成定义为：
      - 电池出口的水，如果没经过电机，直接回到回水混合点
      - 若经过某电机，则用该电机出口温度回到回水混合点
    """
    mdot_total = max(0.0, float(mdot_total_kgps))
    gamma = float(np.clip(gamma_split, 0.0, 1.0))

    mdot_batt = mdot_total

    if mode == 1:
        # 电池独立回路：电池出口直接回到回水混合点；电机无流量
        mdot_f = 0.0
        mdot_r = 0.0
        Tin_f = Tout_batt_degC
        Tin_r = Tout_batt_degC

        # 回水 = 电池出口
        T_return = float(Tout_batt_degC)

    elif mode == 2:
        # 电池出口 -> 前电机；后电机无流量
        mdot_f = mdot_total
        mdot_r = 0.0
        Tin_f = float(Tout_batt_degC)
        Tin_r = float(Tout_batt_degC)

        # 回水 = 前电机出口（因为电池出口都进了前电机）
        T_return = float(Tout_f_degC)

    elif mode == 3:
        # 电池出口 -> 后电机；前电机无流量
        mdot_f = 0.0
        mdot_r = mdot_total
        Tin_f = float(Tout_batt_degC)
        Tin_r = float(Tout_batt_degC)

        T_return = float(Tout_r_degC)

    elif mode == 4:
        # 电池出口 -> 前后并联
        mdot_f = gamma * mdot_total
        mdot_r = (1.0 - gamma) * mdot_total
        Tin_f = float(Tout_batt_degC)
        Tin_r = float(Tout_batt_degC)

        # 回水混合 = (mdot_f*Tout_f + mdot_r*Tout_r) / mdot_total
        if mdot_total > 1e-9:
            T_return = (mdot_f * float(Tout_f_degC) + mdot_r * float(Tout_r_degC)) / mdot_total
        else:
            T_return = float(Tout_batt_degC)

    else:
        raise ValueError("mode must be 1..4")

    return {
        "mdot_batt_kgps": mdot_batt,
        "mdot_f_kgps": mdot_f,
        "mdot_r_kgps": mdot_r,
        "Tin_f_degC": Tin_f,
        "Tin_r_degC": Tin_r,
        "T_return_degC": T_return,
    }
# =========================================================
# 主程序：仿真 + 画图
# =========================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # =========================
    # 0) 文件路径与仿真参数
    # =========================
    dt = 1.0  # 建议与你 BatteryPackModel.dt 一致（你这里 battery 默认 dt=1.0）
    Tamb_degC = -10.0

    speed_csv = "CWTVC.csv"      # time(s), v(km/h)
    cop_xlsx = "COP.xlsx"            # A1空，A列Tamb，第一行Qheat，表内COP

    # 4张电机效率map（CSV）
    f_drive_csv = "Fmotor_Drveff.csv"
    f_brake_csv = "Fmotor_Brkeff.csv"
    r_drive_csv = "Rmotor_Drveff.csv"
    r_brake_csv = "Rmotor_Brkeff.csv"

    # =========================
    # 1) 读取工况与map
    # =========================
    t_prof, v_prof = load_speed_profile_time_v(speed_csv, t_col=0, v_col=1, has_header=True)
    profile = SpeedProfile(t_prof, v_prof)

    front_drive = load_eff_map_csv_A1_empty_Acol_torque_Brow_speed(f_drive_csv, fill_eta=0.2)
    front_brake = load_eff_map_csv_A1_empty_Acol_torque_Brow_speed(f_brake_csv, fill_eta=0.2)
    rear_drive  = load_eff_map_csv_A1_empty_Acol_torque_Brow_speed(r_drive_csv, fill_eta=0.2)
    rear_brake  = load_eff_map_csv_A1_empty_Acol_torque_Brow_speed(r_brake_csv, fill_eta=0.2)

    front_maps = MotorMapSet(front_drive, front_brake)
    rear_maps  = MotorMapSet(rear_drive, rear_brake)

    _, _, _, cop_itp = load_cop_map_excel_A1_empty_Tamb_col_Qrow(cop_xlsx, sheet_name=0, fill_cop=1.0)

    # =========================
    # 2) 初始化模型
    # =========================
    # 车辆/动力学 + 电机
    dyn = DiscreteDualMotorLongitudinalModel(
        dt=dt,
        m=1800.0, Rw=0.33,
        Crr=0.012,
        rho=1.225, A=2.2, Cd=0.29,
        theta=0.0, delta=0.05,
        i_f=9.9, i_r=11.0,
        front_maps=front_maps, rear_maps=rear_maps,
        v0_kmh=profile.v_ref(0.0),
    )

    # 前后电机两状态热模型（用于给出回水温度 Tout_f/Tout_r）
    # 注：R_mc/R_ma 需要你后续标定
    th_f = MotorCoolantTwoStateThermal(dt=dt, R_mc_K_per_W=0.05, R_ma_K_per_W=0.2, Tm0_degC=0.0, Tc0_degC=-5.0)
    th_r = MotorCoolantTwoStateThermal(dt=dt, R_mc_K_per_W=0.05, R_ma_K_per_W=0.2, Tm0_degC=0.0, Tc0_degC=-5.0)

    # 电池模型
    batt = BatteryPackModel(dt=dt, SOC0=0.8, Tb0_degC=-10.0, Tc0_degC=-5.0)
    batt.reset(SOC0=0.8, Tb0_degC=-10.0, Tc0_degC=-5.0)
    #热泵初始化
    hpptc = HeatPumpPTC2L(dt=dt, V_L=2.0, rho_kg_per_m3=1060.0, cp_J_per_kgK=3600.0, T0_degC=-5.0)

    # =========================
    # 3) 外部控制量（后续RL会给）
    # =========================
    mode = 4              # 模式1~4：这里假设全程不变
    alpha_f = 0.3         # 前轴扭矩占比（0~1）
    Kp_speed = 800.0      # 速度跟踪P增益

    mdot_total = 0.2      # kg/s，总泵流量（同时作为电池流量）
    gamma_split = 0.5     # 模式4并联前支路分流比例

    # 热泵/ptc控制（这里先当常数；后续可随时间变化）
    Q_hp_cmd_W = 0.0   # 热泵制热量指令（W，map横轴1000~5000）
    P_ptc_W = 0.0       # PTC电功率（W）

    # =========================
    # 4) 仿真循环
    # =========================
    t_end = profile.t_end
    N = int(np.floor(t_end / dt)) + 1

    # 日志
    t_log = np.zeros(N)
    v_ref_log = np.zeros(N)
    v_act_log = np.zeros(N)

    Tmf_log = np.zeros(N)
    Tmr_log = np.zeros(N)

    Pelec_mot_log = np.zeros(N)     # 电机合计电功率（W）
    Pmot_f_log = np.zeros(N)
    Pmot_r_log = np.zeros(N)
    Pelec_aux_log = np.zeros(N)     # 热泵+PTC电功率（W）
    Pelec_total_log = np.zeros(N)   # 施加给电池的总电功率（W）
    Php_log = np.zeros(N)
    Pptc_log = np.zeros(N)


    Vbatt_log = np.zeros(N)
    Ibatt_log = np.zeros(N)
    SOC_log = np.zeros(N)
    Tb_log = np.zeros(N)
    Tout_batt_log = np.zeros(N)
    Tin_batt_log = np.zeros(N)

    Tm_f_log = np.zeros(N)
    Tm_r_log = np.zeros(N)
    Tin_f_log = np.zeros(N)
    Tin_r_log = np.zeros(N)
    Tout_f_log = np.zeros(N)
    Tout_r_log = np.zeros(N)

    COP_log = np.zeros(N)

    # 初值：回水温度来自“当前状态的各出口”
    # 电池出水 = batt.x[3] (Tc0)；电机出水 = th_*.Tc
    Tout_batt = float(batt.x[3])
    Tout_f = float(th_f.Tc)
    Tout_r = float(th_r.Tc)

    for k in range(N):
        t = k * dt
        t_log[k] = t

        # ---- (1) 参考车速 ----
        v_ref = profile.v_ref(t)
        v_ref_log[k] = v_ref

        # ---- (2) 动力学 + 电机：得到电机电功率 ----
        out_dyn = dyn.step_with_speed_ref(v_ref_kmh=v_ref, alpha_f=alpha_f, Kp=Kp_speed)
        v_act_log[k] = out_dyn["v_kmh"]
        Tmf_log[k] = out_dyn["T_mf_Nm"]
        Tmr_log[k] = out_dyn["T_mr_Nm"]

        Pelec_mot = out_dyn["Pelec_f_W"] + out_dyn["Pelec_r_W"]  # >0耗电，<0回收
        Pelec_mot_log[k] = Pelec_mot
        Pmot_f_log[k] = out_dyn["Pelec_f_W"]
        Pmot_r_log[k] = out_dyn["Pelec_r_W"]

        # ---- (3) 根据模式把水路连起来：算回水混合温度 T_return，以及电机入口/流量 ----
        route = coolant_network_step(
            mode=mode,
            mdot_total_kgps=mdot_total,
            gamma_split=gamma_split,
            Tout_batt_degC=Tout_batt,
            Tout_f_degC=Tout_f,
            Tout_r_degC=Tout_r,
        )

        # ---- (4) 回水 -> 热泵+PTC -> 电池入口温度，同时得到热泵/PTC电耗 ----
        '''
        hp = heatpump_ptc_step_Qcmd(
            T_return_degC=route["T_return_degC"],
            mdot_kgps=route["mdot_batt_kgps"],
            Tamb_degC=Tamb_degC,
            Q_hp_cmd_W=Q_hp_cmd_W,
            P_ptc_elec_W=P_ptc_W,
            cop_itp=cop_itp,
            cp_cool_J_per_kgK=3600.0,
        )
        '''

        hp = hpptc.step(
            T_return_degC=route["T_return_degC"],
            mdot_kgps=route["mdot_batt_kgps"],
            Tamb_degC=Tamb_degC,
            Q_hp_cmd_W=Q_hp_cmd_W,
            P_ptc_elec_W=P_ptc_W,
            cop_itp=cop_itp,
        )

        Tin_batt = hp["T_batt_in_degC"]
        Tin_batt_log[k] = Tin_batt

        P_hp = hp["P_hp_elec_W"]
        P_ptc = hp["P_ptc_elec_W"]

        COP_log[k] = 0.0 if np.isnan(hp["COP"]) else hp["COP"]
        Pptc_log[k] = P_ptc
        Php_log[k] = P_hp

        Pelec_aux = P_hp + P_ptc
        Pelec_aux_log[k] = Pelec_aux

        # ---- (5) 把 电机电耗 + 热泵电耗 + PTC电耗 合在一起施加给电池 ----
        Pelec_total = Pelec_mot + Pelec_aux
        Pelec_total_log[k] = Pelec_total

        # 电池模型是“电流输入”，这里用 I ≈ P/V 估算
        # 注意：你的电池模型电流符号约定是 I>0 => SOC增加（非常规）
        # 因此这里选 I = Pelec_total / Vt，使得耗电(P>0)->I>0->SOC上升（与常规相反，但符合你约定）
        # 后续如果你要把 SOC 物理意义修正，再统一符号即可。
        #Vbatt = batt.peek_voltage(I_A=0.0)
        '''
        Vbatt = max(1e-3, float(batt.x[0]*0 + batt.step(I_A=0.0, Tin_degC=Tin_batt, mdot_kgps=0.0)["Vt_V"]))  # 取一个近似电压
        # 上面这一行是“取当前SOC附近的开路/近似电压”的简化做法；更严谨可以用 batt 当前输出电压或上一步电压
        # 为避免扰动，这里改用上一步记录：
        if k > 0 and Vbatt_log[k-1] > 1e-3:
            Vbatt = Vbatt_log[k-1]
        '''
        if k > 0 and Vbatt_log[k - 1] > 1e-3:
            Vbatt = Vbatt_log[k - 1]
        else:
            Vbatt = max(1e-3, batt.peek_voltage(I_A=0.0))  # 不改变
        I_batt = -Pelec_total / Vbatt

        # 真正更新电池（带流量、入口温度、估算电流）
        y_batt = batt.step(I_A=I_batt, Tin_degC=Tin_batt, mdot_kgps=route["mdot_batt_kgps"])
        Tout_batt = y_batt["Tout_degC"]

        Vbatt_log[k] = y_batt["Vt_V"]
        Ibatt_log[k] = I_batt
        SOC_log[k] = y_batt["SOC"]
        Tb_log[k] = y_batt["Tb_degC"]
        Tout_batt_log[k] = y_batt["Tout_degC"]



        # ---- (6) 更新电机热模型（电机入口温度来自“电池出口”route里已给） ----
        y_f = th_f.step(
            Ploss_W=out_dyn["Ploss_f_W"],
            mdot_kgps=route["mdot_f_kgps"],
            Tin_degC=route["Tin_f_degC"],
            Tamb_degC=Tamb_degC
        )
        y_r = th_r.step(
            Ploss_W=out_dyn["Ploss_r_W"],
            mdot_kgps=route["mdot_r_kgps"],
            Tin_degC=route["Tin_r_degC"],
            Tamb_degC=Tamb_degC
        )
        Tin_f_log[k] = y_f["Tin_degC"]
        Tin_r_log[k] = y_r["Tin_degC"]

        Tout_f = y_f["Tout_degC"]
        Tout_r = y_r["Tout_degC"]

        Tm_f_log[k] = y_f["Tm_degC"]
        Tm_r_log[k] = y_r["Tm_degC"]
        Tout_f_log[k] = y_f["Tout_degC"]
        Tout_r_log[k] = y_r["Tout_degC"]

    # =========================
    # 5) 画图
    # =========================
        # 5) Figure 1（5个子图）

        # =========================

    fig1, ax = plt.subplots(5, 1, figsize=(8, 8), sharex=True)
    ax[0].plot(t_log, v_ref_log, label="v_ref (km/h)", linewidth=2)
    ax[0].plot(t_log, v_act_log, label="v_actual (km/h)", linewidth=2)
    ax[0].set_ylabel("Speed (km/h)")
    ax[0].grid(True)
    ax[0].legend()
    ax[1].plot(t_log, Tmf_log, label="Front motor torque (Nm)")
    ax[1].plot(t_log, Tmr_log, label="Rear motor torque (Nm)")
    ax[1].set_ylabel("Torque (Nm)")
    ax[1].grid(True)
    ax[1].legend()

    ax[2].plot(t_log, Pmot_f_log, label="Front motor elec power (W)")
    ax[2].plot(t_log, Pmot_r_log, label="Rear motor elec power (W)")
    #ax[2].plot(t_log, Pelec_total_log, label="Motor total elec power (W)", linewidth=2)
    ax[2].set_ylabel("Motor P_elec (W)")
    ax[2].grid(True)
    ax[2].legend()

    # 第4子图：PTC与热泵功率（双y轴）

    ax4 = ax[3]
    l1 = ax4.plot(t_log, Php_log, color="tab:blue", label="Heat pump P_elec (W)")
    l2 = ax4.plot(t_log, Pptc_log, color="tab:red", label="PTC P_elec (W)")
    ax4.set_ylabel("HP  and PTC Power (W)")
    ax4.grid(True)
    lines = l1 + l2
    labels = [ln.get_label() for ln in lines]
    ax4.legend(lines, labels, loc="best")
    ax[4].plot(t_log, SOC_log, label="Battery SOC", linewidth=2)
    ax[4].set_xlabel("Time (s)")
    ax[4].set_ylabel("SOC")
    ax[4].grid(True)
    ax[4].legend()

    fig1.tight_layout()
        # =========================

        # 6) Figure 2（4个子图）

        # =========================

    fig2, bx = plt.subplots(4, 1, figsize=(8, 8), sharex=True)
        # (1) 电池电流 + 电压（双y轴）
    b1 = bx[0]
    b1r = b1.twinx()
    l1 = b1.plot(t_log, Ibatt_log, color="tab:blue", label="I_batt (A)")
    l2 = b1r.plot(t_log, Vbatt_log, color="tab:red", label="V_batt (V)")
    b1.set_ylabel("I_batt (A)")
    b1r.set_ylabel("V_batt (V)")
    b1.grid(True)
    lines = l1 + l2
    labels = [ln.get_label() for ln in lines]
    b1.legend(lines, labels, loc="best")

    # (2) 电池温度(右轴) + 出入水口温度(左轴)
    b2 = bx[1]

    l1 = b2.plot(t_log, Tin_batt_log, label="Batt Tin (degC)")
    l2 = b2.plot(t_log, Tout_batt_log, label="Batt Tout (degC)")
    l3 = b2.plot(t_log, Tb_log, color="tab:red", label="Batt Tb (degC)")
    b2.set_ylabel("Motor and Coolant Tm (degC)")
    b2.grid(True)
    lines = l1 + l2 + l3
    labels = [ln.get_label() for ln in lines]
    b2.legend(lines, labels, loc="best")

    # (3) 前电机温度(右轴) + 出入水口(左轴)
    b3 = bx[2]
    l1 = b3.plot(t_log, Tin_f_log, label="Front motor Tin (degC)")
    l2 = b3.plot(t_log, Tout_f_log, label="Front motor Tout (degC)")
    l3 = b3.plot(t_log, Tm_f_log, color="tab:red", label="Front motor Tm (degC)")
    b3.set_ylabel("Motor and Coolant Tm (degC)")
    b3.grid(True)
    lines = l1 + l2 + l3
    labels = [ln.get_label() for ln in lines]
    b3.legend(lines, labels, loc="best")

    # (4) 后电机温度(右轴) + 出入水口(左轴)
    b4 = bx[3]
    l1 = b4.plot(t_log, Tin_r_log, label="Rear motor Tin (degC)")
    l2 = b4.plot(t_log, Tout_r_log, label="Rear motor Tout (degC)")
    l3 = b4.plot(t_log, Tm_r_log, color="tab:red", label="Rear motor Tm (degC)")
    b4.set_xlabel("Time (s)")
    b4.set_ylabel("Motor and Coolant Tm (degC)")
    b4.grid(True)
    lines = l1 + l2 + l3
    labels = [ln.get_label() for ln in lines]
    b4.legend(lines, labels, loc="best")
    fig2.tight_layout()

    plt.show()