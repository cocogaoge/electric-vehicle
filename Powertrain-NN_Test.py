import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers

# =========================
# 0) 路径设置
# =========================
TEST_DATA_DIR = r"D:\powertrain_model\Test_data"
OUT_DIR = r"D:\powertrain_model\Train_NN_model\NN_Result"
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_DIRS = {
    "MLP": r"D:\powertrain_model\Train_NN_model\PureNN_t_to_t1_out_MLP",
    "LSTM": r"D:\powertrain_model\Train_NN_model\PureNN_t_to_t1_out_LSTM",
    "GRU": r"D:\powertrain_model\Train_NN_model\PureNN_t_to_t1_out_GRU",
    "TRANSFORMER": r"D:\powertrain_model\Train_NN_model\PureNN_t_to_t1_out_TRANSFORMER",
}

TARGETS = ["P_elec_tot", "P_Fmotor", "P_Rmotor", "Tout_front", "Tout_rear"]

# 你给的映射
RENAME_MAPPING = {
    "ts": "ts",
    "vehspd": "v_tgt",
    # "BrkSysAxRoadSlope": "Slope",
    "vculgtslopfordisp": "Slope",
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

# =========================
# 1) 自定义层：TransformerBlock
# =========================
class TransformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            layers.Dense(d_ff, activation="relu"),
            layers.Dense(d_model),
        ])
        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(dropout)
        self.drop2 = layers.Dropout(dropout)

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = dropout

    def call(self, x, training=False):
        attn = self.mha(x, x, training=training)
        x = self.ln1(x + self.drop1(attn, training=training))
        f = self.ffn(x, training=training)
        x = self.ln2(x + self.drop2(f, training=training))
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "d_ff": self.d_ff,
            "dropout": self.dropout,
        })
        return cfg

# =========================
# 2) 工具函数
# =========================
def norm01(x, mn, mx):
    return (x - mn) / (mx - mn + 1e-8)

def denorm01(xn, mn, mx):
    return xn * (mx - mn + 1e-8) + mn

def rmse_mae(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))
    mae = np.mean(np.abs(y_true - y_pred), axis=0)
    return rmse, mae

def make_windows(X_t, seq_len=32, stride=1):
    N = X_t.shape[0]
    if N < seq_len:
        return None
    end_idx = np.arange(seq_len - 1, N, stride)
    M = len(end_idx)
    Xw = np.empty((M, seq_len, X_t.shape[1]), dtype=np.float32)
    for k, i in enumerate(end_idx):
        Xw[k] = X_t[i - seq_len + 1: i + 1]
    return Xw

def safe_predict_numpy(model, X, batch_size=4096):
    if X is None or len(X) == 0:
        return None
    ds = tf.data.Dataset.from_tensor_slices(X).batch(batch_size)
    return model.predict(ds, verbose=0)

def load_norm_params(model_dir):
    npz = np.load(os.path.join(model_dir, "norm_params.npz"), allow_pickle=True)
    return (
        npz["x_min"], npz["x_max"],
        npz["y_min"], npz["y_max"],
        list(npz["features"]),
        list(npz["targets"]),
    )

def load_one_model(model_dir):
    model_path = os.path.join(model_dir, "pure_nn_final.keras")
    return tf.keras.models.load_model(
        model_path,
        custom_objects={"TransformerBlock": TransformerBlock},
        compile=False
    )

def read_data_csv(csv_path, needed_cols):
    """只对 needed_cols 做 to_numeric + dropna，避免无关脏列导致全空。"""
    df = pd.read_csv(csv_path)
    df.rename(columns=RENAME_MAPPING, inplace=True)
    df = df.loc[:, ~df.columns.duplicated(keep="first")]

    exist = [c for c in needed_cols if c in df.columns]
    df[exist] = df[exist].apply(pd.to_numeric, errors="coerce")

    # derive power
    if ("P_Fmotor" not in df.columns) and {"Cur_Fmotor", "U_Fmotor"}.issubset(df.columns):
        df["P_Fmotor"] = 0.001 * df["Cur_Fmotor"] * df["U_Fmotor"]
    if ("P_Rmotor" not in df.columns) and {"Cur_Rmotor", "U_Rmotor"}.issubset(df.columns):
        df["P_Rmotor"] = 0.001 * df["Cur_Rmotor"] * df["U_Rmotor"]
    if ("P_elec_tot" not in df.columns) and {"P_Fmotor", "P_Rmotor"}.issubset(df.columns):
        df["P_elec_tot"] = df["P_Fmotor"] + df["P_Rmotor"]

    # derive cooling
    if ("mdot_sum" not in df.columns) and {"mdot_front", "mdot_rear"}.issubset(df.columns):
        df["mdot_sum"] = df["mdot_front"] + df["mdot_rear"]
    if ("dT_front" not in df.columns) and {"Tin_front", "Tout_front"}.issubset(df.columns):
        df["dT_front"] = df["Tout_front"] - df["Tin_front"]
    if ("dT_rear" not in df.columns) and {"Tin_front", "Tout_rear"}.issubset(df.columns):
        df["dT_rear"] = df["Tout_rear"] - df["Tin_front"]

    df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    subset = [c for c in needed_cols if c in df.columns]
    df = df.dropna(subset=subset)
    return df

# =========================
# 3) 对齐绘图（关键：x轴用 ts[1:]，序列模型从 ts[seq_len:]）
# =========================
def plot_timeseries_aligned(ts_y, out_df, file_stem, out_dir):
    """
    ts_y: 对齐到 t+1 的时间轴（等长于 out_df）
    out_df: 包含 True_*, MLP_*, LSTM_*, GRU_*, TRANSFORMER_* （部分为 NaN 是正常的）
    """
    for t in TARGETS:
        plt.figure(figsize=(14, 4))
        plt.plot(ts_y, out_df[f"True_{t}"], label="True", linewidth=2, color="black")
        for m in ["MLP", "LSTM", "GRU", "TRANSFORMER"]:
            col = f"{m}_{t}"
            if col in out_df.columns:
                plt.plot(ts_y, out_df[col], label=m, linewidth=1)
        plt.title(f"{file_stem} | {t} | aligned to t+1")
        plt.xlabel("ts (aligned)")
        plt.ylabel(t)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{file_stem}__{t}__aligned.png"), dpi=150)
        plt.close()

# =========================
# 4) 主流程
# =========================
def main():
    pack = {}
    all_needed = set(TARGETS)

    for name, mdir in MODEL_DIRS.items():
        x_min, x_max, y_min, y_max, features, targets = load_norm_params(mdir)
        model = load_one_model(mdir)
        pack[name] = {
            "model": model,
            "x_min": x_min, "x_max": x_max,
            "y_min": y_min, "y_max": y_max,
            "features": features,
            "targets": targets
        }
        all_needed.update(features)
        print(f"[Loaded] {name}: features={len(features)}, targets={targets}")

    # 派生需要的基础列
    all_needed.update(["Cur_Fmotor", "U_Fmotor", "Cur_Rmotor", "U_Rmotor",
                       "Tin_front", "Tout_front", "Tout_rear",
                       "mdot_front", "mdot_rear",
                       "ts"])  # 为了画图对齐

    test_files = sorted(glob.glob(os.path.join(TEST_DATA_DIR, "*.csv")))
    if not test_files:
        raise RuntimeError(f"No csv found in {TEST_DATA_DIR}")

    summary_rows = []

    for csv_path in test_files:
        file_stem = os.path.splitext(os.path.basename(csv_path))[0]
        print(f"\n=== Testing: {file_stem} ===")

        df = read_data_csv(csv_path, needed_cols=list(all_needed))
        if len(df) < 2:
            print(f"[Skip] {file_stem}: too short after clean (len={len(df)})")
            continue

        # 时间轴：对齐到 t+1
        if "ts" in df.columns:
            ts = df["ts"].to_numpy()
            ts_y = ts[1:]  # 与 Y_true 对齐
        else:
            ts_y = np.arange(len(df) - 1)

        if not set(TARGETS).issubset(df.columns):
            miss = [c for c in TARGETS if c not in df.columns]
            print(f"[Skip] {file_stem}: missing targets {miss}")
            continue

        Y_true = df[TARGETS].to_numpy(np.float32)[1:]
        out_df = pd.DataFrame({f"True_{t}": Y_true[:, i] for i, t in enumerate(TARGETS)})

        # 四模型预测
        for name in ["MLP", "LSTM", "GRU", "TRANSFORMER"]:
            model = pack[name]["model"]
            features = pack[name]["features"]
            x_min = pack[name]["x_min"]
            x_max = pack[name]["x_max"]
            y_min = pack[name]["y_min"]
            y_max = pack[name]["y_max"]

            miss_f = [c for c in features if c not in df.columns]
            if miss_f:
                print(f"[Skip model] {file_stem} {name}: missing features {miss_f[:10]} ...")
                continue

            X = df[features].to_numpy(np.float32)

            if name == "MLP":
                Xn = norm01(X, x_min, x_max)
                X_in = Xn[:-1]                      # t=0..N-2
                Yhat_n = safe_predict_numpy(model, X_in)
                if Yhat_n is None:
                    print(f"[Skip model] {file_stem} {name}: no samples")
                    continue
                Yhat = denorm01(Yhat_n, y_min, y_max)  # 对齐到 t+1: 1..N-1

                for i, t in enumerate(TARGETS):
                    out_df[f"{name}_{t}"] = Yhat[:, i]

                rmse, mae = rmse_mae(Y_true, Yhat)

            else:
                seq_len = model.input_shape[1]
                Xn2 = norm01(X, x_min, x_max)
                Xw = make_windows(Xn2, seq_len=seq_len, stride=1)
                if Xw is None:
                    print(f"[Skip model] {file_stem} {name}: too short for seq_len={seq_len}")
                    continue

                Yhat_n = safe_predict_numpy(model, Xw)
                if Yhat_n is None:
                    print(f"[Skip model] {file_stem} {name}: predict None")
                    continue
                Yhat = denorm01(Yhat_n, y_min, y_max)

                # out_df 对应 ts[1:]，其 index 0 是原始 t=1
                # 序列模型第一个预测对应原始 t=seq_len （因为 window end at seq_len-1 -> predict seq_len）
                start = seq_len - 1  # out_df 的 start index（对齐到 ts[1:]）
                pred_len = Yhat.shape[0]

                for i, t in enumerate(TARGETS):
                    col = np.full((len(out_df),), np.nan, dtype=np.float32)
                    end = min(start + pred_len, len(out_df))
                    col[start:end] = Yhat[:end - start, i]
                    out_df[f"{name}_{t}"] = col

                valid_true = Y_true[start:start + pred_len]
                valid_pred = Yhat[:len(valid_true)]
                rmse, mae = rmse_mae(valid_true, valid_pred)

            row = {"file": file_stem, "model": name}
            for i, t in enumerate(TARGETS):
                row[f"RMSE_{t}"] = float(rmse[i])
                row[f"MAE_{t}"] = float(mae[i])
            summary_rows.append(row)

        # 保存对比 CSV（对齐到 t+1）
        out_csv = os.path.join(OUT_DIR, f"{file_stem}__pred_compare_aligned.csv")
        out_df.insert(0, "ts_y", ts_y)  # 把对齐后的时间轴也写进去
        out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

        # 对齐绘图（关键修复）
        plot_timeseries_aligned(ts_y, out_df, file_stem, OUT_DIR)
        print(f"Saved: {out_csv} and aligned plots (*.png)")

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(OUT_DIR, "summary_metrics_by_file_and_model.csv")
        summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
        print(f"\nAll done. Summary saved to: {summary_path}")
    else:
        print("\nNo valid results produced (all files skipped?).")

if __name__ == "__main__":
    main()