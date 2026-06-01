# ============================================================
# Pure NN baseline (t -> t+1, ONE model, 5 outputs)
# Switchable architectures: MLP / LSTM / GRU / TRANSFORMER
#
# - Features from Corr_*.csv + FORCE_FEATURES
# - Rename CAN columns -> standard names (RENAME_MAPPING)
# - Derived cols: P_Fmotor/P_Rmotor/P_elec_tot, mdot_sum, dT_front, dT_rear
# - MLP uses pointwise input X[t] -> Y[t+1]
# - LSTM/GRU/TRANSFORMER use sequence window:
#       X[t-SEQ_LEN+1 : t]  ->  Y[t+1]
# ============================================================

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
from tqdm import tqdm

# =========================
# 0) Settings (EDIT HERE)
# =========================
SEED = 40

MODEL_TYPE = "LSTM"          # "MLP" / "LSTM" / "GRU" / "TRANSFORMER"
SEQ_LEN = 32                # used for LSTM/GRU/TRANSFORMER
SEQ_STRIDE = 1              # used for LSTM/GRU/TRANSFORMER

THRESHOLD = 0.2
MODE = "UNION4"             # "ALL"/"DRIVE"/"REGEN"/"COOL_ON"/"UNION4"

BATCH = 4096
EPOCHS = 100
LR = 3e-4

MAX_ROWS_TRAIN = None
MAX_ROWS_VAL = None

TARGETS = ["P_elec_tot", "P_Fmotor", "P_Rmotor", "Tout_front", "Tout_rear"]
FORCE_FEATURES = ["Slope", "Load", "SOC", "Brk_masterCylinder_pressure", "Brk_Cylinder_pressure"]

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
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))  # ...\Train_NN_model
PROJECT_DIR = os.path.dirname(CURRENT_DIR)                # ...\powertrain_model

DATA_ROOT = os.path.join(PROJECT_DIR, "Train_data")
LIST_TRAIN = os.path.join(DATA_ROOT, "train_files.txt")
LIST_VAL   = os.path.join(DATA_ROOT, "val_files.txt")

CORR_DIR = os.path.join(CURRENT_DIR, "Corr")

OUT_DIR = os.path.join(CURRENT_DIR, f"PureNN_t_to_t1_out_{MODEL_TYPE}")
os.makedirs(OUT_DIR, exist_ok=True)

print("MODEL_TYPE  =", MODEL_TYPE)
print("SEQ_LEN     =", SEQ_LEN if MODEL_TYPE != "MLP" else None)
print("SEQ_STRIDE  =", SEQ_STRIDE if MODEL_TYPE != "MLP" else None)
print("CURRENT_DIR =", CURRENT_DIR)
print("PROJECT_DIR =", PROJECT_DIR)
print("DATA_ROOT   =", DATA_ROOT)
print("CORR_DIR    =", CORR_DIR)
print("OUT_DIR     =", OUT_DIR)

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
# 4) Correlation-based features + forced features
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
FEATURES = [c for c in FEATURES if c not in TARGETS]  # prevent leakage
FEATURES = sorted(set(FEATURES).union(FORCE_FEATURES))

print("Selected FEATURES (raw + forced):", FEATURES)
print("Num FEATURES (raw + forced):", len(FEATURES))

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
print("Train files:", len(train_files))
print("Val files:", len(val_files))

def read_data_csv(csv_path):
    df = pd.read_csv(csv_path)

    # rename
    df.rename(columns=RENAME_MAPPING, inplace=True)

    # remove duplicated columns after rename
    df = df.loc[:, ~df.columns.duplicated(keep="first")]

    # numeric
    df = df.apply(pd.to_numeric, errors="coerce")

    # derive power kW
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

    return df

# sanity check
tmp_raw = pd.read_csv(train_files[0], nrows=1)
print("RAW cols sample:", list(tmp_raw.columns)[:25])
tmp = read_data_csv(train_files[0])
print("RENAMED cols sample:",
      [c for c in tmp.columns if c in (TARGETS + FORCE_FEATURES + ["v_tgt","Tin_front","mdot_sum","dT_front","dT_rear"])])

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

FEATURES_RAW = FEATURES[:]
FEATURES = [c for c in FEATURES if c in existing_cols]

missing_targets = [t for t in TARGETS if t not in existing_cols]
if missing_targets:
    raise RuntimeError(f"Targets missing in training CSV columns: {missing_targets}")

print("FEATURES before filter:", FEATURES_RAW)
print("Final FEATURES after filter:", FEATURES)
print("Num FEATURES after filter:", len(FEATURES))

if len(FEATURES) == 0:
    raise RuntimeError("After filtering, no FEATURES left.")

# =========================
# 7) Build dataset: t -> t+1
# =========================
def clean_needed(df, needed_cols):
    d = df[needed_cols].copy()
    d = d.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    d = d.dropna()
    return d

def build_xy_pointwise_t_to_t1(df, feature_cols, target_cols):
    needed = list(dict.fromkeys(feature_cols + target_cols))
    if any(c not in df.columns for c in needed):
        return None, None
    d = clean_needed(df, needed)
    if len(d) < 3:
        return None, None
    X = d[feature_cols].to_numpy(np.float32)
    Y = d[target_cols].to_numpy(np.float32)
    return X[:-1], Y[1:]  # X[t] -> Y[t+1]

def make_windows_for_t_to_t1(X_t, Y_t1, seq_len=32, stride=1):
    """
    X_t  shape: (N, F)  time t = 0..N-1
    Y_t1 shape: (N, O)  label aligned with X_t at same index (meaning Y[t+1])
    window ending at index i -> label Y_t1[i]
    """
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
    skipped_missing = 0
    skipped_short = 0

    for fp in tqdm(files, desc=f"Collecting dataset ({model_type})"):
        df = read_data_csv(fp)
        X_t, Y_t1 = build_xy_pointwise_t_to_t1(df, feature_cols, target_cols)
        if X_t is None:
            needed = list(dict.fromkeys(feature_cols + target_cols))
            if any(c not in df.columns for c in needed):
                skipped_missing += 1
            else:
                skipped_short += 1
            continue

        if model_type == "MLP":
            Xs.append(X_t); Ys.append(Y_t1)
            total += len(X_t)
        else:
            Xw, Yw = make_windows_for_t_to_t1(X_t, Y_t1, seq_len=SEQ_LEN, stride=SEQ_STRIDE)
            if Xw is None:
                skipped_short += 1
                continue
            Xs.append(Xw); Ys.append(Yw)
            total += len(Xw)

        if (max_rows is not None) and (total >= max_rows):
            break

    print(f"[collect_dataset] skipped_missing_cols={skipped_missing}, skipped_short_or_nan={skipped_short}")

    if not Xs:
        raise RuntimeError("No valid samples built.")

    Xall = np.concatenate(Xs, axis=0)
    Yall = np.concatenate(Ys, axis=0)

    if (max_rows is not None) and (len(Xall) > max_rows):
        Xall = Xall[:max_rows]
        Yall = Yall[:max_rows]
    return Xall, Yall

if MODEL_TYPE not in ("MLP", "LSTM", "GRU", "TRANSFORMER"):
    raise ValueError("MODEL_TYPE must be one of: MLP/LSTM/GRU/TRANSFORMER")

X_train, Y_train = collect_dataset(train_files, FEATURES, TARGETS, MODEL_TYPE, max_rows=MAX_ROWS_TRAIN)
X_val,   Y_val   = collect_dataset(val_files,   FEATURES, TARGETS, MODEL_TYPE, max_rows=MAX_ROWS_VAL)

print("X_train:", X_train.shape, "Y_train:", Y_train.shape)
print("X_val:",   X_val.shape,   "Y_val:",   Y_val.shape)

# =========================
# 8) Normalization (fit on train)
# =========================
def norm01(x, mn, mx):
    return (x - mn) / (mx - mn + 1e-8)

def denorm01(xn, mn, mx):
    return xn * (mx - mn + 1e-8) + mn

if MODEL_TYPE == "MLP":
    x_min = np.min(X_train, axis=0)
    x_max = np.max(X_train, axis=0)
    X_train_n = norm01(X_train, x_min, x_max)
    X_val_n   = norm01(X_val,   x_min, x_max)
else:
    x_min = np.min(X_train.reshape(-1, X_train.shape[-1]), axis=0)
    x_max = np.max(X_train.reshape(-1, X_train.shape[-1]), axis=0)
    X_train_n = norm01(X_train, x_min[None, None, :], x_max[None, None, :])
    X_val_n   = norm01(X_val,   x_min[None, None, :], x_max[None, None, :])

y_min = np.min(Y_train, axis=0)
y_max = np.max(Y_train, axis=0)
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
    inp = layers.Input(shape=(n_in,), name="X_t")
    x = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(1e-5))(inp)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(1e-5))(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-5))(x)
    out = layers.Dense(n_out, activation="sigmoid", name="Y_t1_norm01")(x)
    return Model(inp, out, name="PureNN_MLP_t_to_t1")

def build_rnn(cell="LSTM", seq_len=32, n_feat=10, n_out=5):
    inp = layers.Input(shape=(seq_len, n_feat), name="X_seq")
    if cell == "LSTM":
        x = layers.LSTM(128, return_sequences=True)(inp)
        x = layers.LSTM(64)(x)
    else:
        x = layers.GRU(128, return_sequences=True)(inp)
        x = layers.GRU(64)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(n_out, activation="sigmoid", name="Y_t1_norm01")(x)
    return Model(inp, out, name=f"PureNN_{cell}_t_to_t1")

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

    def call(self, x, training=False):
        attn = self.mha(x, x, training=training)
        x = self.ln1(x + self.drop1(attn, training=training))
        f = self.ffn(x, training=training)
        x = self.ln2(x + self.drop2(f, training=training))
        return x

def build_transformer(seq_len, n_feat, n_out,
                      d_model=64, num_heads=4, d_ff=128, n_blocks=2, dropout=0.1):
    inp = layers.Input(shape=(seq_len, n_feat), name="X_seq")
    x = layers.Dense(d_model)(inp)

    pos = tf.range(start=0, limit=seq_len, delta=1)
    pos_emb = layers.Embedding(input_dim=seq_len, output_dim=d_model)(pos)
    x = x + pos_emb[None, :, :]

    for i in range(n_blocks):
        x = TransformerBlock(d_model, num_heads, d_ff, dropout, name=f"tf_block_{i}")(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(n_out, activation="sigmoid", name="Y_t1_norm01")(x)
    return Model(inp, out, name="PureNN_Transformer_t_to_t1")

# choose model
if MODEL_TYPE == "MLP":
    model = build_mlp(n_in=X_train_n.shape[1], n_out=len(TARGETS))
elif MODEL_TYPE == "LSTM":
    model = build_rnn(cell="LSTM", seq_len=X_train_n.shape[1], n_feat=X_train_n.shape[2], n_out=len(TARGETS))
elif MODEL_TYPE == "GRU":
    model = build_rnn(cell="GRU", seq_len=X_train_n.shape[1], n_feat=X_train_n.shape[2], n_out=len(TARGETS))
else:  # TRANSFORMER
    model = build_transformer(seq_len=X_train_n.shape[1], n_feat=X_train_n.shape[2], n_out=len(TARGETS))

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=[tf.keras.metrics.MeanAbsoluteError(name="MAE")]
)

callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1),
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(OUT_DIR, "pure_nn_best.keras"),
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
]

# =========================
# 11) Train
# =========================
history = model.fit(ds_train, validation_data=ds_val, epochs=EPOCHS, callbacks=callbacks, verbose=2)

# =========================
# 12) Save artifacts
# =========================
model.save(os.path.join(OUT_DIR, "pure_nn_final.keras"))
pd.DataFrame(history.history).to_csv(os.path.join(OUT_DIR, "history.csv"), index=False)

meta = {
    "seed": SEED,
    "model_type": MODEL_TYPE,
    "seq_len": SEQ_LEN if MODEL_TYPE != "MLP" else None,
    "seq_stride": SEQ_STRIDE if MODEL_TYPE != "MLP" else None,
    "threshold": THRESHOLD,
    "mode": MODE,
    "task": "X[t] -> Y[t+1]" if MODEL_TYPE == "MLP" else "X[t-SEQ_LEN+1:t] -> Y[t+1]",
    "features": FEATURES,
    "forced_features": FORCE_FEATURES,
    "targets": TARGETS
}
pd.Series({k: str(v) for k, v in meta.items()}).to_csv(os.path.join(OUT_DIR, "meta.csv"))

np.savez(
    os.path.join(OUT_DIR, "norm_params.npz"),
    x_min=x_min, x_max=x_max,
    y_min=y_min, y_max=y_max,
    features=np.array(FEATURES, dtype=object),
    forced_features=np.array(FORCE_FEATURES, dtype=object),
    targets=np.array(TARGETS, dtype=object)
)

# =========================
# 13) Evaluate (original units)
# =========================
Yhat_val_n = model.predict(X_val_n, batch_size=BATCH, verbose=0)
Yhat_val = denorm01(Yhat_val_n, y_min, y_max)

rmse = np.sqrt(np.mean((Yhat_val - Y_val) ** 2, axis=0))
mae  = np.mean(np.abs(Yhat_val - Y_val), axis=0)

report = pd.DataFrame({"target": TARGETS, "RMSE": rmse, "MAE": mae})
report.to_csv(os.path.join(OUT_DIR, "val_report.csv"), index=False)

print("Done. Saved to:", OUT_DIR)
print(report)