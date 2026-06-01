import os, glob
import numpy as np
import pandas as pd

# ========= 1) 你需要改的路径 =========
TRAIN_DATA_DIR = r"D:\powertrain_model\Train_NN_model\NN_Result_Temp\Selected_Cases\Top5pct_BestFiles_RawCSV"
# 上面这个目录里应该是训练用 CSV（你训练脚本里 DATA_ROOT 指向的 Train_data）

MODEL_DIRS = {
    "GRU": r"D:\powertrain_model\Train_NN_model\Run_PURE_GRU_ToutFrontRear_dt",
    "LSTM": r"D:\powertrain_model\Train_NN_model\Run_PURE_LSTM_ToutFrontRear_dt",
    "MLP": r"D:\powertrain_model\Train_NN_model\Run_PURE_MLP_ToutFrontRear_dt",
    "TRANSFORMER": r"D:\powertrain_model\Train_NN_model\Run_PURE_TRANSFORMER_ToutFrontRear_dt",
}

# 训练目标（与你训练脚本一致）
TARGETS = ["Tout_front", "Tout_rear"]

# ========= 2) 你的列名映射（保持一致） =========
RENAME_MAPPING = {
    "ts": "ts",
    "vehspd": "v_tgt",
    "vculgtslopfordisp": "Slope",
    "BrkSysAxRoadSlope": "Slope",
    "prsntloaddata": "Load",
    "HvBattActBusU": "Ubat",
    "HvBattActCur": "Ibat",
    "BattSocRaw": "SOC",
    "FrntMotActSpd": "Spd_Fmotor",
    "FrntMotActTq": "Trq_Fmotor",
    "FrntMotActCur": "Cur_Fmotor",
    "FrntMotActU": "U_Fmotor",
    "ReMotActSpd": "Spd_Rmotor",
    "ReMotActTq": "Trq_Rmotor",
    "ReMotActCur": "Cur_Rmotor",
    "ReMotActU": "U_Rmotor",
    "AgsaActPosn": "ags_open",
    "HvBattInletCooltT": "Tin_front",
    "FrntMotCooltT": "Tout_front",
    "ReMotCooltT": "Tout_rear",
    "FrntMotMotT": "T_Fmotor",
    "ReMotMotT": "T_Rmotor",
    "FrntMotCooltCircFlowEstimd": "mdot_front",
    "ReMotCooltCircFlowEstimd": "mdot_rear",
    "hvbattoutletcooltt": "Tout_Batt",
    "bcpmstcylp": "Brk_masterCylinder_pressure",
    "mstcylpcmp": "Brk_Cylinder_pressure",
}

DT_FRONT = "dT_front"
DT_REAR  = "dT_rear"

def read_data_csv(csv_path):
    df = pd.read_csv(csv_path)
    df.rename(columns=RENAME_MAPPING, inplace=True)
    df = df.loc[:, ~df.columns.duplicated(keep="first")]
    df = df.apply(pd.to_numeric, errors="coerce")

    # dT 特征（和训练一致：Tout - Tin）
    if ("Tout_front" in df.columns) and ("Tin_front" in df.columns):
        df[DT_FRONT] = df["Tout_front"] - df["Tin_front"]
    if ("Tout_rear" in df.columns) and ("Tin_front" in df.columns):
        df[DT_REAR] = df["Tout_rear"] - df["Tin_front"]

    df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    return df

def infer_features_from_weights(model_dir):
    """
    关键点：你目录里没有 FEATURES 列表。
    但你的训练脚本最终 FEATURES 是固定的（你日志里打印过 Num FEATURES=24）。
    这里给一个“兜底固定列表”，与你日志一致。
    如果你后续 FEATURES 有变化，请在这里同步更新。
    """
    FEATURES = [
        "Cur_Fmotor", "Cur_Rmotor", "Ibat", "Load", "SOC", "Slope",
        "Spd_Fmotor", "Spd_Rmotor", "T_Fmotor", "T_Rmotor",
        "Tin_front", "Tout_Batt", "Tout_front", "Tout_rear",
        "Trq_Fmotor", "Trq_Rmotor", "U_Fmotor", "U_Rmotor",
        "ags_open", "dT_front", "dT_rear", "mdot_front", "mdot_rear", "v_tgt"
    ]
    return FEATURES

def update_minmax(cur_min, cur_max, x):
    mn = np.nanmin(x, axis=0)
    mx = np.nanmax(x, axis=0)
    if cur_min is None:
        return mn, mx
    return np.minimum(cur_min, mn), np.maximum(cur_max, mx)

def build_norm_params_for_dir(model_dir, train_files, features, targets):
    x_min = x_max = y_min = y_max = None

    need = list(dict.fromkeys(features + targets))
    for fp in train_files:
        df = read_data_csv(fp)
        if any(c not in df.columns for c in need):
            continue
        d = df[need].dropna()
        if len(d) < 3:
            continue

        X = d[features].to_numpy(np.float32)
        Y = d[targets].to_numpy(np.float32)

        x_min, x_max = update_minmax(x_min, x_max, X)
        y_min, y_max = update_minmax(y_min, y_max, Y)

    if x_min is None:
        raise RuntimeError(f"No valid training data found to compute min/max for: {model_dir}")

    out_npz = os.path.join(model_dir, "norm_params.npz")
    np.savez(
        out_npz,
        x_min=x_min, x_max=x_max,
        y_min=y_min, y_max=y_max,
        features=np.array(features, dtype=object),
        targets=np.array(targets, dtype=object),
    )
    print("[DONE] Saved:", out_npz)

def main():
    train_files = sorted(glob.glob(os.path.join(TRAIN_DATA_DIR, "*.csv")))
    if not train_files:
        raise RuntimeError(f"No train csv found in {TRAIN_DATA_DIR}")

    for name, mdir in MODEL_DIRS.items():
        if not os.path.exists(os.path.join(mdir, "final_backbone.keras")):
            raise FileNotFoundError(f"Missing final_backbone.keras in {mdir}")

        features = infer_features_from_weights(mdir)
        build_norm_params_for_dir(mdir, train_files, features, TARGETS)

if __name__ == "__main__":
    main()
