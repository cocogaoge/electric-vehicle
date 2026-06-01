import numpy as np
import gymnasium as gym

from thermalvehicle import coolant_network_step


class ThermoDriveEnv(gym.Env):
    #环境搭建函数
    metadata = {"render_modes": []}

    def __init__(
        self,
        profile,#速度曲线
        dyn,#纵向动力学，带电机
        batt,#电池系统电热耦合
        th_f,#前电机热模型
        th_r,#后电机热模型
        hpptc,  #热泵及PTC系统        # HeatPumpPTC2L instance
        cop_itp, #热泵空调的COP插值器
        dt=1.0, #步长
        Tamb_degC=10.0,#环境温度
        gamma_split=0.5,     # 模式4下，前后电机流量分配比例，固定1:1并联
        Kp_speed=800.0, #纵向动力学驾驶员模型K
        # targets
        Tbatt_tar=5.0, #电池目标温度
        Tm_tar=0.0, #电机目标温度
        # reward weights
        wE=0.02,            # 能耗权重（对kW）
        wsw=0.5,            # mode切换惩罚
    ):
        super().__init__()
        self.profile = profile
        self.dyn = dyn
        self.batt = batt
        self.th_f = th_f
        self.th_r = th_r
        self.hpptc = hpptc
        self.cop_itp = cop_itp

        self.dt = float(dt)
        self.Tamb = float(Tamb_degC)
        self.gamma_split = float(gamma_split)
        self.Kp_speed = float(Kp_speed)

        self.Tbatt_tar = float(Tbatt_tar)
        self.Tm_tar = float(Tm_tar)

        self.wE = float(wE)
        self.wsw = float(wsw)

        # TD3连续动作
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32) #连续动作-1~1

        # obs维度（你可以按需增减）
        self.obs_dim = 1 + 1 + 1 + 4 + 6 + 2  # e_v + v + v_ref + battery(4) + motors(6) + (mode,mdot)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)

        self.reset()

    def _decode_action(self, a):
        #将动作-1~1解码到各执行器实际范围
        #ptc0~3000，热泵0~5000，模式1~4，流量0~12L/min需要进一步转化为kgs，前后扭矩分配比例0~1
        RHO_EG50 = 1060.0
        a = np.asarray(a, dtype=float)

        P_ptc = 3000.0 * (a[0] + 1.0) / 2.0
        Q_hp  = 5000.0 * (a[1] + 1.0) / 2.0

        z = (a[2] + 1.0) / 2.0
        mode = int(1 + min(3, np.floor(4.0 * z)))

        Vdot_Lmin = 12.0 * (a[3] + 1.0) / 2.0
        mdot = RHO_EG50 * (Vdot_Lmin * 1e-3 / 60.0)

        alpha_f = (a[4] + 1.0) / 2.0
        return float(P_ptc), float(Q_hp), mode, float(mdot), float(alpha_f)

    @staticmethod
    def _temp_asym_penalty(T, Ttar, w_low=1.0, w_high=0.2):
        #温度的不对称奖励，因为是加热，所以温度低惩罚大，温度高一点惩罚小一点
        low = max(0.0, Ttar - T)
        high = max(0.0, T - Ttar)
        return w_low * low * low + w_high * high * high

    def _get_obs(self):
        #获取观测量，车速误差，真实车速，参考车速，SOC，电池温度，电池出入口温度，前后电机出入口温度，模式，流量
        # 当前时间点参考速度
        v_ref = self.profile.v_ref(self.t)
        v = self.dyn.v * 3.6
        e_v = v_ref - v

        SOC, Tb, _, Tout_batt = self.batt.x  # batt.x=[SOC,Tb,Up,Tc]，Tout=Tc
        Tin_batt = self.Tin_batt

        # 电机
        Tm_f, Tin_f, Tout_f = self.Tm_f, self.Tin_f, self.Tout_f
        Tm_r, Tin_r, Tout_r = self.Tm_r, self.Tin_r, self.Tout_r

        # 归一化（可调整）
        mdot_max = 1060.0 * (12.0 * 1e-3 / 60.0)

        obs = np.array([
            e_v / 30.0,
            v / 120.0,
            v_ref / 120.0,
            SOC,
            Tb / 50.0,
            Tin_batt / 50.0,
            Tout_batt / 50.0,
            Tm_f / 50.0, Tin_f / 50.0, Tout_f / 50.0,
            Tm_r / 50.0, Tin_r / 50.0, Tout_r / 50.0,
            (self.mode - 1) / 3.0,
            self.mdot / mdot_max if mdot_max > 0 else 0.0,
        ], dtype=np.float32)
        return obs

    def reset(self, seed=None, options=None):
        #初始状态随机化在这里改
        super().reset(seed=seed)

        self.t = 0.0
        self.k = 0
        self.mode = 1
        self.mdot = 0.0

        # 这里做温度随机化（后续可按需要改范围）
        rng = self.np_random
        Tb0 = float(rng.uniform(-10.0, 5.0))
        Tc_b0 = float(rng.uniform(-10.0, 5.0))
        Tm0 = float(rng.uniform(-10.0, 5.0))
        Tc_m0 = float(rng.uniform(-10.0, 5.0))
        Thp0 = float(rng.uniform(-10.0, 5.0))
        SOC0 = float(rng.uniform(0.4, 0.9))

        self.dyn.reset(v0_kmh=self.profile.v_ref(0.0))
        self.batt.reset(SOC0=SOC0, Tb0_degC=Tb0, Tc0_degC=Tc_b0)
        self.th_f.reset(Tm0_degC=Tm0, Tc0_degC=Tc_m0)
        self.th_r.reset(Tm0_degC=Tm0, Tc0_degC=Tc_m0)
        self.hpptc.reset(T0_degC=Thp0)

        # 保存用于水路计算的“出口温度状态”（用当前模型内部的出口）
        self.Tout_batt = float(self.batt.x[3])
        self.Tout_f = float(self.th_f.Tc)
        self.Tout_r = float(self.th_r.Tc)

        # 入口温度占位（用于obs）
        self.Tin_batt = float(self.hpptc.T)
        self.Tin_f = self.Tout_batt
        self.Tin_r = self.Tout_batt

        self.Tm_f = float(self.th_f.Tm)
        self.Tm_r = float(self.th_r.Tm)

        self.prev_mode = self.mode
        self.V_prev = self.batt.peek_voltage(0.0)
        return self._get_obs(), {}

    def step(self, action):
        #步进函数
        P_ptc, Q_hp, mode, mdot, alpha_f = self._decode_action(action)

        self.prev_mode = self.mode
        self.mode = mode
        self.mdot = mdot

        # 1) 参考车速
        v_ref = self.profile.v_ref(self.t)

        # 2) 动力学 + 电机（P控制生成总轮端扭矩）
        out_dyn = self.dyn.step_with_speed_ref(v_ref_kmh=v_ref, alpha_f=alpha_f, Kp=self.Kp_speed)
        Pelec_mot = out_dyn["Pelec_f_W"] + out_dyn["Pelec_r_W"]

        # 3) 水路模式：mode=1 电机流量为0，但电池流量=mdot（你确认）
        route = coolant_network_step(
            mode=self.mode,
            mdot_total_kgps=self.mdot,
            gamma_split=self.gamma_split,
            Tout_batt_degC=self.Tout_batt,
            Tout_f_degC=self.Tout_f,
            Tout_r_degC=self.Tout_r,
        )
        self.Tin_f = float(route["Tin_f_degC"])
        self.Tin_r = float(route["Tin_r_degC"])

        # 4) 回水 -> 热泵+PTC(2L) -> 电池入口
        hp = self.hpptc.step(
            T_return_degC=float(route["T_return_degC"]),
            mdot_kgps=float(route["mdot_batt_kgps"]),
            Tamb_degC=self.Tamb,
            Q_hp_cmd_W=Q_hp,
            P_ptc_elec_W=P_ptc,
            cop_itp=self.cop_itp,
        )
        self.Tin_batt = float(hp["T_batt_in_degC"])
        Pelec_aux = float(hp["P_hp_elec_W"] + hp["P_ptc_elec_W"])

        # 5) 电池：总电功率施加到电池
        Pelec_total = Pelec_mot + Pelec_aux
        # I ≈ P/V (用上一步电压；这里用当前状态估一个)
        #V_est = max(1e-3, 350.0)  # 你也可以用上一步V，或用 batt.step 输出的V
        V_est = max(1e-3, self.V_prev)
        I_batt = -Pelec_total / V_est

        y_batt = self.batt.step(I_A=I_batt, Tin_degC=self.Tin_batt, mdot_kgps=float(route["mdot_batt_kgps"]))
        self.Tout_batt = float(y_batt["Tout_degC"])
        SOC = float(y_batt["SOC"])
        Tbatt = float(y_batt["Tb_degC"])
        self.V_prev = y_batt["Vt_V"]

        # 6) 电机热
        y_f = self.th_f.step(out_dyn["Ploss_f_W"], float(route["mdot_f_kgps"]), self.Tin_f, self.Tamb)
        y_r = self.th_r.step(out_dyn["Ploss_r_W"], float(route["mdot_r_kgps"]), self.Tin_r, self.Tamb)
        self.Tm_f = float(y_f["Tm_degC"]); self.Tout_f = float(y_f["Tout_degC"])
        self.Tm_r = float(y_r["Tm_degC"]); self.Tout_r = float(y_r["Tout_degC"])

        # 7) reward：温度目标 + 能耗 + mode切换
        pen_T = (
            self._temp_asym_penalty(Tbatt, self.Tbatt_tar, w_low=1.0, w_high=0.2)
            + self._temp_asym_penalty(self.Tm_f, self.Tm_tar, w_low=0.5, w_high=0.2)
            + self._temp_asym_penalty(self.Tm_r, self.Tm_tar, w_low=0.5, w_high=0.2)
        )
        # 能耗惩罚（kW尺度）；只惩罚耗电
        pen_E = self.wE * max(Pelec_total, 0.0) / 1000.0
        pen_sw = self.wsw * (1.0 if self.mode != self.prev_mode else 0.0)

        reward = -(pen_T + pen_E + pen_sw)

        # 8) episode结束：整条工况结束
        self.k += 1
        self.t += self.dt
        terminated = (self.t >= self.profile.t_end)
        truncated = False

        info = {
            "Pelec_total_W": Pelec_total,
            "Pelec_mot_W": Pelec_mot,
            "Pelec_aux_W": Pelec_aux,
            "mode": self.mode,
            "mdot": self.mdot,
            "Tin_batt": self.Tin_batt,
            "Tout_batt": self.Tout_batt,
        }

        return self._get_obs(), float(reward), terminated, truncated, info

