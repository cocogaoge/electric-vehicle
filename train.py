import os
import time

from env import ThermoDriveEnv
from thermalvehicle import load_speed_profile_time_v, SpeedProfile, load_eff_map_csv_A1_empty_Acol_torque_Brow_speed, \
    MotorMapSet, load_cop_map_excel_A1_empty_Tamb_col_Qrow, DiscreteDualMotorLongitudinalModel, BatteryPackModel, \
    MotorCoolantTwoStateThermal, HeatPumpPTC2L
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

def sanity_check_env(env, n_steps=200, seed=0):
    #这个函数是一个环境检测函数，输入已经搭建好的环境，调用这个函数可以查看在随机动作下的环境反馈
    obs, info = env.reset(seed=seed)
    print("reset ok | obs shape:", obs.shape, "dtype:", obs.dtype)
    ep_ret = 0.0
    for k in range(n_steps):
        a = env.action_space.sample()#采样随机动作
        obs, r, terminated, truncated, info = env.step(a)#获取随机动作后的状态
        ep_ret += r
        if (k % 20) == 0 or terminated or truncated:
            print(
                f"[{k:04d}] r={r: .4f} ep_ret={ep_ret: .2f} "
                f"mode={info.get('mode')} mdot={info.get('mdot'): .4f} "
                f"Ptot={info.get('Pelec_total_W'): .1f}W "
                f"Tin_batt={info.get('Tin_batt'): .2f} Tout_batt={info.get('Tout_batt'): .2f}"
            )
        if not np.all(np.isfinite(obs)):
            raise RuntimeError(f"Non-finite obs at step {k}: {obs}")
        if not np.isfinite(r):
            raise RuntimeError(f"Non-finite reward at step {k}: {r}")

        if terminated or truncated:
            print("episode finished:", {"terminated": terminated, "truncated": truncated})
            break
    print("done | total return:", ep_ret)

def make_env(
    speed_csv="CWTVC.csv",
    cop_xlsx="COP.xlsx",
    dt=1.0,
    Tamb_degC=-10.0,
    f_drive_csv="Fmotor_Drveff.csv",
    f_brake_csv="Fmotor_Brkeff.csv",
    r_drive_csv="Rmotor_Drveff.csv",
    r_brake_csv="Rmotor_Brkeff.csv",
):

    # 1) profile
    t_prof, v_prof = load_speed_profile_time_v(speed_csv, t_col=0, v_col=1, has_header=True)
    profile = SpeedProfile(t_prof, v_prof)
    # 2) motor maps
    front_drive = load_eff_map_csv_A1_empty_Acol_torque_Brow_speed(f_drive_csv, fill_eta=0.2)
    front_brake = load_eff_map_csv_A1_empty_Acol_torque_Brow_speed(f_brake_csv, fill_eta=0.2)
    rear_drive = load_eff_map_csv_A1_empty_Acol_torque_Brow_speed(r_drive_csv, fill_eta=0.2)
    rear_brake = load_eff_map_csv_A1_empty_Acol_torque_Brow_speed(r_brake_csv, fill_eta=0.2)
    front_maps = MotorMapSet(front_drive, front_brake)
    rear_maps = MotorMapSet(rear_drive, rear_brake)
    # 3) dynamics
    dyn = DiscreteDualMotorLongitudinalModel(
        dt=dt,
        m=1800.0,
        Rw=0.31,
        Crr=0.012,
        rho=1.225,
        A=2.2,
        Cd=0.29,
        theta=0.0,
        delta=0.05,
        i_f=9.9,
        i_r=11.0,
        front_maps=front_maps,
        rear_maps=rear_maps,
        v0_kmh=profile.v_ref(0.0),
    )


    # 4) thermal
    batt = BatteryPackModel(dt=dt, Tamb_degC=Tamb_degC)
    th_f = MotorCoolantTwoStateThermal(dt=dt)
    th_r = MotorCoolantTwoStateThermal(dt=dt)

    # 5) hp+ptc + cop map
    hpptc = HeatPumpPTC2L(dt=dt)
    _, _, _, cop_itp = load_cop_map_excel_A1_empty_Tamb_col_Qrow(cop_xlsx, sheet_name=0, fill_cop=1.0)

    # 6) env

    env = ThermoDriveEnv(
        profile=profile,
        dyn=dyn,
        batt=batt,
        th_f=th_f,
        th_r=th_r,
        hpptc=hpptc,
        cop_itp=cop_itp,
        dt=dt,
        Tamb_degC=Tamb_degC,
    )
    return env

if __name__ == "__main__":

    # ---------- 训练输出目录 ----------
    run_dir = "runs_td3"
    tb_root = os.path.join(run_dir, "tb_runs")  # 根目录
    os.makedirs(tb_root, exist_ok=True)
    tb_name = time.strftime("TD3_%Y%m%d_%H%M%S")
    tb_dir = os.path.join(tb_root, tb_name)  # 本次训练专属目录
    os.makedirs(tb_dir, exist_ok=True)
    print("TensorBoard log dir:", os.path.abspath(tb_dir))



    ckpt_dir = os.path.join(run_dir, "checkpoints")
    best_dir = os.path.join(run_dir, "best_model")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(best_dir, exist_ok=True)

    # ---------- VecEnv ----------
    # 单环境也建议用 VecEnv（SB3要求）
    train_env = DummyVecEnv([lambda: make_env(Tamb_degC=-10.0, dt=1.0)])
    # 观测/奖励归一化（TD3强烈建议，尤其你reward是负的惩罚项叠加）
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
    )

    # 评估环境：必须单独建一份，并且评估时用训练得到的归一化参数
    eval_env = DummyVecEnv([lambda: make_env(Tamb_degC=-10.0, dt=1.0)])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,  # 评估时一般看原始reward更直观；也可以 True
        clip_obs=10.0,
    )
    eval_env.training = False
    eval_env.norm_reward = False

    # ---------- TD3探索噪声 ----------
    # 你的动作是 [-1,1]，5维；噪声可先用 0.1~0.2 级别
    n_actions = train_env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.15 * np.ones(n_actions))

    # ---------- 回调：保存checkpoint + 保存best ----------
    checkpoint_cb = CheckpointCallback(
        save_freq=50_000,  # 每多少 step 保存一次
        save_path=ckpt_dir,
        name_prefix="td3",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=best_dir,
        log_path=os.path.join(run_dir, "eval_logs"),
        eval_freq=20_000,
        n_eval_episodes=3,
        deterministic=True,
        render=False,
    )

    # ---------- TD3模型 ----------
    model = TD3(
        policy="MlpPolicy",
        env=train_env,
        action_noise=action_noise,
        learning_rate=3e-4,
        buffer_size=300_000,
        learning_starts=10_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=(1, "step"),
        gradient_steps=1,
        policy_delay=2,
        target_policy_noise=0.2,
        target_noise_clip=0.5,
        verbose=1,
        tensorboard_log=None,
        device="auto",
    )

    # ---------- 开始训练 ----------
    total_timesteps = 500_000  # 你先跑 50万看看趋势；后续可上百万
    model.learn(total_timesteps=total_timesteps, callback=[checkpoint_cb, eval_cb])
    # ---------- 保存最终模型 + VecNormalize ----------
    final_path = os.path.join(run_dir, "td3_final")
    model.save(final_path)
    train_env.save(os.path.join(run_dir, "vecnormalize.pkl"))

    print("Training finished. Saved to:", final_path)

#测试代码
'''
if __name__ == "__main__":
    speed_csv = "CWTVC.csv"  # time(s), v(km/h)
    cop_xlsx = "COP.xlsx"  # A1空，A列Tamb，第一行Qheat，表内COP
    dt = 1
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
    rear_drive = load_eff_map_csv_A1_empty_Acol_torque_Brow_speed(r_drive_csv, fill_eta=0.2)
    rear_brake = load_eff_map_csv_A1_empty_Acol_torque_Brow_speed(r_brake_csv, fill_eta=0.2)

    front_maps = MotorMapSet(front_drive, front_brake)
    rear_maps = MotorMapSet(rear_drive, rear_brake)

    # ---- 3) 纵向动力学
    dyn = DiscreteDualMotorLongitudinalModel(
    dt=dt,
    m=1800.0,
    Rw=0.31,
    Crr=0.012,
    rho=1.225,
    A=2.2,
    Cd=0.29,
    theta=0.0,
    delta=0.05,
    i_f=9.9,
    i_r=11.0,
    front_maps=front_maps,
    rear_maps=rear_maps,
    v0_kmh=profile.v_ref(0.0),
    )

    # ---- 4) 电池/电机热模型
    batt = BatteryPackModel(dt=dt, Tamb_degC=10.0)
    th_f = MotorCoolantTwoStateThermal(dt=dt)
    th_r = MotorCoolantTwoStateThermal(dt=dt)
    # ---- 5) 热泵+PTC 2L
    hpptc = HeatPumpPTC2L(dt=dt)
    _, _, _, cop_itp = load_cop_map_excel_A1_empty_Tamb_col_Qrow(cop_xlsx, sheet_name=0, fill_cop=1.0)

    env = ThermoDriveEnv(
        profile=profile,
        dyn=dyn,
        batt=batt,
        th_f=th_f,
        th_r=th_r,
        hpptc=hpptc,
        cop_itp=cop_itp,
        dt=dt,
        Tamb_degC=-10.0,
    )
    sanity_check_env(env, n_steps=500, seed=42)
'''