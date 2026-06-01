# ============================================================
# Switchable: MLP / LSTM / GRU / TRANSFORMER
# Switchable training: PURE / PINN_B
# Task:
#   MLP:            X[t] -> [Tout_front, Tout_rear][t+1]
#   LSTM/GRU/Trans: X[t-L+1:t] -> [Tout_front, Tout_rear][t+1]
#
# Predict 2 targets:
#   [Tout_front, Tout_rear] at t+1
#
# Extra features added:
#   dT_front = Tout_front - Tin_front
#   dT_rear  = Tout_rear  - Tin_front   (Tin_rear = Tin_front)
#
# PINN_B constraints:
#   - rear inlet uses Tin_front
#   - mdot_front/mdot_rear are non-negative
#   (1) Cooling consistency:
#       Tout_front >= Tin_front when mdot_front > 0
#       Tout_rear  >= Tin_front when mdot_rear  > 0
#       Penalty: relu(Tin - Tout) * mdot
#
# NOTE (server-safe saving):
#   - ModelCheckpoint saves weights only (best.weights.h5)
#   - Final save saves backbone only (final_backbone.keras)
# ============================================================

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
from tqdm import tqdm

import keras
print("TF:", tf.__version__)
print("keras:", keras.__version__)

# =========================
# 0) Settings (EDIT HERE)
# =========================
SEED = 40

MODEL_TYPE = "MLP"        # "MLP" / "LSTM" / "GRU" / "TRANSFORMER"
TRAIN_MODE = "PURE"       # "PURE" / "PINN_B"

SEQ_LEN = 32
SEQ_STRIDE = 1

THRESHOLD = 0.2
MODE = "UNION4"

BATCH = 4096
EPOCHS = 1000
LR = 1e-3

# PINN weights
LAMBDA_COOL_SIGN = 0.1
LAMBDA_SMOOTH = 0.0  # off by default

MAX_ROWS_TRAIN = None
MAX_ROWS_VAL = None

TARGETS = ["Tout_front", "Tout_rear"]

# 新增温差特征名
DT_FRONT = "dT_front"   # Tout_front - Tin_front
DT_REAR  = "dT_rear"    # Tout_rear  - Tin_front  (推荐方向)

# 如果您“必须”用 Tin_front - Tout_rear，把下面设为 True
USE_TIN_MINUS_TOUT_REAR = False

# 强制特征（可按需要调整）
FORCE_FEATURES = [
    "v_tgt", "Slope", "Load", "SOC",
    "Spd_Fmotor", "Trq_Fmotor", "Cur_Fmotor", "U_Fmotor",
    "Spd_Rmotor", "Trq_Rmotor", "Cur_Rmotor", "U_Rmotor",
    "T_Fmotor", "T_Rmotor",
    "Tout_front", "Tout_rear",
    DT_FRONT, DT_REAR,   # <<<<<< 加在这里
]

# PINN 必需输入
PINN_REQUIRED_FEATURES = ["Tin_front", "mdot_front", "mdot_rear"]

# =========================
# 1) Reproducibility
# =========================
def set_seed(seed=40):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"

set_seed(SEED)

# =========================
# 2) Relative paths
# =========================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)

DATA_ROOT = os.path.join(PROJECT_DIR, "Train_data")
LIST_TRAIN = os.path.join(DATA_ROOT, "train_files.txt")
LIST_VAL   = os.path.join(DATA_ROOT, "val_files.txt")

CORR_DIR = os.path.join(CURRENT_DIR, "Corr")

OUT_DIR = os.path.join(CURRENT_DIR, f"Run_{TRAIN_MODE}_{MODEL_TYPE}_ToutFrontRear")
os.makedirs(OUT_DIR, exist_ok=True)

print("TRAIN_MODE =", TRAIN_MODE)
print("MODEL_TYPE =", MODEL_TYPE)
print("SEQ_LEN    =", SEQ_LEN if MODEL_TYPE != "MLP" else None)
print("OUT_DIR    =", OUT_DIR)

# =========================
# 3) Rename mapping
# =========================
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

# =========================
# 4) Correlation-based features + forced + PINN required
# =========================
def load_corr(csv_path):
    return pd.read_csv(csv_path, index_col=0)

def extract_key_factors(corr_df, target_var, threshold=0.2):
    s = corr_df[target_var].copy()
    s = s.drop(index=[target_var], errors="ignore")
    return s.index[(s.abs() > threshold)].tolist()

def features_for_target(target, threshold=0.2, mode="UNION4"):
    corr_all  = load_corr(os.path.join(CORR_DIR, "Corr_ALL.csv"))
    corr_drv  = load_corr(os.path.join(CORR_DIR, "Corr_DRIVE.csv"))
    corr_reg  = load_corr(os.path.join(CORR_DIR, "Corr_REGEN.csv"))
    corr_cool = load_corr(os.path.join(CORR_DIR, "Corr_COOL_ON.csv"))

    if mode == "ALL":
        feats = extract_key_factors(corr_all, target, threshold)
    elif mode == "DRIVE":
        feats = extract_key_factors(corr_drv, target, threshold)
    elif mode == "REGEN":
        feats = extract_key_factors(corr_reg, target, threshold)
    elif mode == "COOL_ON":
        feats = extract_key_factors(corr_cool, target, threshold)
    elif mode == "UNION4":
        feats = sorted(set(
            extract_key_factors(corr_all, target, threshold) +
            extract_key_factors(corr_drv, target, threshold) +
            extract_key_factors(corr_reg, target, threshold) +
            extract_key_factors(corr_cool, target, threshold)
        ))
    else:
        raise ValueError("Unknown mode")
    return feats

FEATURES = sorted(set().union(*[features_for_target(t, THRESHOLD, MODE) for t in TARGETS]))
FEATURES = [c for c in FEATURES if c not in TARGETS]
FEATURES = sorted(set(FEATURES).union(FORCE_FEATURES))
if TRAIN_MODE == "PINN_B":
    FEATURES = sorted(set(FEATURES).union(PINN_REQUIRED_FEATURES))

print("FEATURES (raw):", FEATURES)

# =========================
# 5) IO helpers
# =========================
def load_file_list(list_path, root_dir):
    files = []
    with open(list_path, "r", encoding="utf-8") as f:
        for line in f:
            rel = line.strip()
            if rel:
                files.append(os.path.join(root_dir, rel))
    return [p for p in files if os.path.exists(p)]

train_files = load_file_list(LIST_TRAIN, DATA_ROOT)
val_files   = load_file_list(LIST_VAL,   DATA_ROOT)
print("Train files:", len(train_files), "Val files:", len(val_files))

def read_data_csv(csv_path):
    """
    关键修改点：在这里计算 dT_front / dT_rear，然后就能参与 FEATURES 选择与训练。
    """
    df = pd.read_csv(csv_path)
    df.rename(columns=RENAME_MAPPING, inplace=True)
    df = df.loc[:, ~df.columns.duplicated(keep="first")]
    df = df.apply(pd.to_numeric, errors="coerce")

    # ===== 新增：计算温差特征 =====
    # dT_front = Tout_front - Tin_front
    if ("Tout_front" in df.columns) and ("Tin_front" in df.columns):
        df[DT_FRONT] = df["Tout_front"] - df["Tin_front"]

    # dT_rear = Tout_rear - Tin_front  (推荐方向)
    if ("Tout_rear" in df.columns) and ("Tin_front" in df.columns):
        if USE_TIN_MINUS_TOUT_REAR:
            df[DT_REAR] = df["Tin_front"] - df["Tout_rear"]   # 您要求的方向
        else:
            df[DT_REAR] = df["Tout_rear"] - df["Tin_front"]   # 推荐方向（出口-入口）
    # ============================

    return df

# =========================
# 6) Filter FEATURES by existing columns
# =========================
def get_existing_columns(sample_files, max_check=50):
    cols = set()
    n = 0
    for fp in sample_files:
        try:
            df = read_data_csv(fp)
            cols |= set(df.columns)
            n += 1
            if n >= max_check:
                break
        except Exception:
            continue
    return cols

existing_cols = get_existing_columns(train_files, max_check=50)
FEATURES = [c for c in FEATURES if c in existing_cols]

missing_targets = [t for t in TARGETS if t not in existing_cols]
if missing_targets:
    raise RuntimeError(f"Targets missing: {missing_targets}")

if TRAIN_MODE == "PINN_B":
    missing_pinn = [c for c in PINN_REQUIRED_FEATURES if c not in existing_cols]
    if missing_pinn:
        raise RuntimeError(f"PINN required features missing: {missing_pinn}")

print("Final FEATURES:", FEATURES)
print("Num FEATURES:", len(FEATURES))

# =========================
# 7) Dataset building (t -> t+1)
# =========================
def clean_needed(df, cols):
    d = df[cols].copy()
    d = d.replace([np.inf, -np.inf], np.nan).ffill().bfill().dropna()
    return d

def build_xy_pointwise_t_to_t1(df, feature_cols, target_cols):
    need = list(dict.fromkeys(feature_cols + target_cols))
    if any(c not in df.columns for c in need):
        return None, None
    d = clean_needed(df, need)
    if len(d) < 3:
        return None, None
    X = d[feature_cols].to_numpy(np.float32)
    Y = d[target_cols].to_numpy(np.float32)
    return X[:-1], Y[1:]  # t -> t+1

def make_windows(X_t, Y_t1, seq_len=32, stride=1):
    N = X_t.shape[0]
    if N < seq_len:
        return None, None
    end_idx = np.arange(seq_len - 1, N, stride)
    M = len(end_idx)
    Xw = np.empty((M, seq_len, X_t.shape[1]), dtype=np.float32)
    Yw = np.empty((M, Y_t1.shape[1]), dtype=np.float32)
    for k, i in enumerate(end_idx):
        Xw[k] = X_t[i - seq_len + 1: i + 1]
        Yw[k] = Y_t1[i]
    return Xw, Yw

def collect_dataset(files, feature_cols, target_cols, model_type, max_rows=None):
    Xs, Ys = [], []
    total = 0
    for fp in tqdm(files, desc=f"Collect ({model_type})"):
        df = read_data_csv(fp)
        X_t, Y_t1 = build_xy_pointwise_t_to_t1(df, feature_cols, target_cols)
        if X_t is None:
            continue

        if model_type == "MLP":
            Xs.append(X_t); Ys.append(Y_t1)
            total += len(X_t)
        else:
            Xw, Yw = make_windows(X_t, Y_t1, seq_len=SEQ_LEN, stride=SEQ_STRIDE)
            if Xw is None:
                continue
            Xs.append(Xw); Ys.append(Yw)
            total += len(Xw)

        if max_rows is not None and total >= max_rows:
            break

    if not Xs:
        raise RuntimeError("No valid samples built.")

    Xall = np.concatenate(Xs, axis=0)
    Yall = np.concatenate(Ys, axis=0)
    if max_rows is not None and len(Xall) > max_rows:
        Xall = Xall[:max_rows]
        Yall = Yall[:max_rows]
    return Xall, Yall

X_train, Y_train = collect_dataset(train_files, FEATURES, TARGETS, MODEL_TYPE, max_rows=MAX_ROWS_TRAIN)
X_val,   Y_val   = collect_dataset(val_files,   FEATURES, TARGETS, MODEL_TYPE, max_rows=MAX_ROWS_VAL)

print("X_train:", X_train.shape, "Y_train:", Y_train.shape)

# =========================
# 8) Normalization
# =========================
def norm01(x, mn, mx):
    return (x - mn) / (mx - mn + 1e-8)

def denorm01(xn, mn, mx):
    return xn * (mx - mn + 1e-8) + mn

if MODEL_TYPE == "MLP":
    x_min = np.min(X_train, axis=0); x_max = np.max(X_train, axis=0)
    X_train_n = norm01(X_train, x_min, x_max)
    X_val_n   = norm01(X_val,   x_min, x_max)
else:
    x_min = np.min(X_train.reshape(-1, X_train.shape[-1]), axis=0)
    x_max = np.max(X_train.reshape(-1, X_train.shape[-1]), axis=0)
    X_train_n = norm01(X_train, x_min[None, None, :], x_max[None, None, :])
    X_val_n   = norm01(X_val,   x_min[None, None, :], x_max[None, None, :])

y_min = np.min(Y_train, axis=0); y_max = np.max(Y_train, axis=0)
Y_train_n = norm01(Y_train, y_min, y_max)
Y_val_n   = norm01(Y_val,   y_min, y_max)

# =========================
# 9) tf.data
# =========================
AUTOTUNE = tf.data.AUTOTUNE
ds_train = tf.data.Dataset.from_tensor_slices((X_train_n, Y_train_n)).shuffle(200_000, seed=SEED).batch(BATCH).prefetch(AUTOTUNE)
ds_val   = tf.data.Dataset.from_tensor_slices((X_val_n,   Y_val_n)).batch(BATCH).prefetch(AUTOTUNE)

# =========================
# 10) Models
# =========================
def build_mlp(n_in, n_out):
    inp = layers.Input(shape=(n_in,), name="X")
    x = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(1e-5))(inp)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(1e-5))(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-5))(x)
    out = layers.Dense(n_out, activation="sigmoid", name="Yhat")(x)
    return Model(inp, out, name="MLP")

def build_rnn(cell="LSTM", seq_len=32, n_feat=10, n_out=2):
    inp = layers.Input(shape=(seq_len, n_feat), name="Xseq")
    if cell == "LSTM":
        x = layers.LSTM(128, return_sequences=True)(inp)
        x = layers.LSTM(64)(x)
    else:
        x = layers.GRU(128, return_sequences=True)(inp)
        x = layers.GRU(64)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(n_out, activation="sigmoid", name="Yhat")(x)
    return Model(inp, out, name=cell)

class TransformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([layers.Dense(d_ff, activation="relu"), layers.Dense(d_model)])
        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(dropout)
        self.drop2 = layers.Dropout(dropout)

    def call(self, x, training=False):
        attn = self.mha(x, x, training=training)
        x = self.ln1(x + self.drop1(attn, training=training))
        f = self.ffn(x, training=training)
        x = self.ln2(x + self.drop2(f, training=training))
        return x

def build_transformer(seq_len, n_feat, n_out, d_model=64, num_heads=4, d_ff=128, n_blocks=2, dropout=0.1):
    inp = layers.Input(shape=(seq_len, n_feat), name="Xseq")
    x = layers.Dense(d_model)(inp)
    pos = tf.range(start=0, limit=seq_len, delta=1)
    pos_emb = layers.Embedding(input_dim=seq_len, output_dim=d_model)(pos)
    x = x + pos_emb[None, :, :]
    for i in range(n_blocks):
        x = TransformerBlock(d_model, num_heads, d_ff, dropout, name=f"blk{i}")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(n_out, activation="sigmoid", name="Yhat")(x)
    return Model(inp, out, name="TRANSFORMER")

# =========================
# 11) PINN trainer
# =========================
target_index = {name: i for i, name in enumerate(TARGETS)}
feat_index = {name: i for i, name in enumerate(FEATURES)}

idx_ToutF = target_index["Tout_front"]
idx_ToutR = target_index["Tout_rear"]

idx_Tin = feat_index.get("Tin_front", None)
idx_mf  = feat_index.get("mdot_front", None)
idx_mr  = feat_index.get("mdot_rear", None)

class PINNTrainer(tf.keras.Model):
    def __init__(self, model_config, **kwargs):
        super().__init__(**kwargs)
        self.model_config = model_config
        self.backbone = self.build_model_from_config(model_config)
        self.loss_mse = tf.keras.losses.MeanSquaredError()

    def build_model_from_config(self, model_config):
        model_type = model_config["type"]
        n_out = 2
        if model_type == "MLP":
            return build_mlp(n_in=model_config["n_in"], n_out=n_out)
        elif model_type == "LSTM":
            return build_rnn(cell="LSTM", seq_len=model_config["seq_len"], n_feat=model_config["n_feat"], n_out=n_out)
        elif model_type == "GRU":
            return build_rnn(cell="GRU", seq_len=model_config["seq_len"], n_feat=model_config["n_feat"], n_out=n_out)
        else:
            return build_transformer(seq_len=model_config["seq_len"], n_feat=model_config["n_feat"], n_out=n_out)

    def call(self, x, training=False):
        return self.backbone(x, training=training)

    def train_step(self, data):
        x, y_true = data
        with tf.GradientTape() as tape:
            y_pred = self.backbone(x, training=True)
            L_data = self.loss_mse(y_true, y_pred)

            if TRAIN_MODE == "PINN_B":
                if MODEL_TYPE == "MLP":
                    Tin = x[:, idx_Tin]
                    mf  = x[:, idx_mf]
                    mr  = x[:, idx_mr]
                else:
                    Tin = x[:, -1, idx_Tin]
                    mf  = x[:, -1, idx_mf]
                    mr  = x[:, -1, idx_mr]

                viol_f = tf.nn.relu(Tin - y_pred[:, idx_ToutF]) * mf
                viol_r = tf.nn.relu(Tin - y_pred[:, idx_ToutR]) * mr
                L_cool = tf.reduce_mean(viol_f + viol_r)

                loss = L_data + LAMBDA_COOL_SIGN * L_cool + LAMBDA_SMOOTH * 0.0
            else:
                loss = L_data

        grads = tape.gradient(loss, self.backbone.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.backbone.trainable_variables))

        err = y_true - y_pred
        mae = tf.reduce_mean(tf.abs(err))
        rmse = tf.sqrt(tf.reduce_mean(tf.square(err)))
        return {"loss": loss, "data_loss": L_data, "MAE": mae, "RMSE": rmse}

    def test_step(self, data):
        x, y_true = data
        y_pred = self.backbone(x, training=False)
        L_data = self.loss_mse(y_true, y_pred)
        err = y_true - y_pred
        mae = tf.reduce_mean(tf.abs(err))
        rmse = tf.sqrt(tf.reduce_mean(tf.square(err)))
        return {"loss": L_data, "MAE": mae, "RMSE": rmse}

model_config = {
    "type": MODEL_TYPE,
    "n_in": X_train_n.shape[1] if MODEL_TYPE == "MLP" else None,
    "seq_len": SEQ_LEN,
    "n_feat": len(FEATURES),
}

trainer = PINNTrainer(model_config)
trainer.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR))
trainer.backbone.summary()

# =========================
# 11.5) Callbacks (FIXED: save weights only)
# =========================
best_w_path = os.path.join(OUT_DIR, "best.weights.h5")

callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_RMSE",
        mode="min",
        patience=30,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        best_w_path,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    ),
]

# 强制 build（避免 epoch1 保存时报 not yet been built）
for xb, yb in ds_train.take(1):
    _ = trainer(xb, training=False)

history = trainer.fit(ds_train, validation_data=ds_val, epochs=EPOCHS, callbacks=callbacks, verbose=2)

if os.path.exists(best_w_path):
    trainer.load_weights(best_w_path)

# =========================
# 12) Evaluate (original units)
# =========================
def denorm01(xn, mn, mx):
    return xn * (mx - mn + 1e-8) + mn

Yhat_val_n = trainer.predict(X_val_n, batch_size=BATCH, verbose=0)
Yhat_val = denorm01(Yhat_val_n, y_min, y_max)

rmse = np.sqrt(np.mean((Yhat_val - Y_val) ** 2, axis=0))
mae  = np.mean(np.abs(Yhat_val - Y_val), axis=0)

report = pd.DataFrame({"target": TARGETS, "RMSE": rmse, "MAE": mae})
report.to_csv(os.path.join(OUT_DIR, "val_report.csv"), index=False)

# =========================
# 13) Save (server-safe)
# =========================
trainer.backbone.save(os.path.join(OUT_DIR, "final_backbone.keras"))
trainer.save_weights(os.path.join(OUT_DIR, "final.weights.h5"))

pd.DataFrame(history.history).to_csv(os.path.join(OUT_DIR, "history.csv"), index=False)

print("Done. Saved to:", OUT_DIR)
print(report)