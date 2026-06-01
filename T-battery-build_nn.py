import numpy as np
from sympy.series import sequences
from tensorflow.keras import layers, Model, optimizers, regularizers
from tensorflow.keras.models import load_model
import tensorflow as tf
from read_battery_data import *
import glob  # 用于查找文件路径
import os    # 用于拼接路径
import random
from tqdm import tqdm
from tf_battery import *
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('TkAgg') # <--- 新增此行，强制使用 TkAgg 后端
from keras import ops

def set_seed(seed=42):
    """
    设置所有相关的随机种子，以确保实验的可复现性。
    """
    # 设置 Python 内置的随机种子
    random.seed(seed)
    # 设置 NumPy 的随机种子
    np.random.seed(seed)
    # 设置 TensorFlow 的随机种子
    tf.random.set_seed(seed)
    # 强制 TensorFlow 使用确定性的操作
    # 这可能会对性能产生轻微影响，但对于可复现性至关重要
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    # (可选) 设置 Python 哈希种子
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"所有随机种子已固定为: {seed}")
# --- 在所有其他代码执行之前调用此函数 ---
SEED_VALUE = 40
set_seed(SEED_VALUE)

#def build_parameter_nn(input_shape):
#    inputs = layers.Input(shape=input_shape)
#    x = layers.Dense(128, activation='relu')(inputs)
#    x = layers.Dense(128, activation='relu')(x)
#    x = layers.Dense(64, activation='relu')(x)
#    # 输出3个参数，使用softplus确保它们是正数
#    # R_o, R_p, C_p 物理上必须为正，还有换热系数h和液冷板热阻Rc
#    outputs = layers.Dense(5, activation=tf.nn.softplus)(x)
#    model = Model(inputs, outputs)
#    return model

def build_parameter_nn(input_shape):
    inputs = layers.Input(shape=input_shape, name='nn_inputs')
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    # 最后一层输出 raw（无激活）
    raw = layers.Dense(5, activation=None, name='param_raw')(x)
    # 参数范围（按需修改）
    rom_l, rom_u = 1e-4, 2     # Rom
    rp_l,  rp_u  = 1e-4, 2.0      # Rp
    cp_l,  cp_u  = 1e-4, 1e5      # Cp（跨度大，用对数映射）
    h_l,   h_u   = 0.1, 100.0     # h
    rc_l,  rc_u  = 1e-4, 100.0    # Rc（跨度大，用对数映射）
    # 把 raw 映射到指定区间
    def map_params(r):
        s = tf.nn.sigmoid(r)
        rom = rom_l + (rom_u - rom_l) * s[..., 0]
        rp  = rp_l  + (rp_u  - rp_l)  * s[..., 1]
        cp  = tf.exp(tf.math.log(cp_l) + (tf.math.log(cp_u) - tf.math.log(cp_l)) * s[..., 2])
        h   = h_l   + (h_u   - h_l)   * s[..., 3]
        rc  = tf.exp(tf.math.log(rc_l) + (tf.math.log(rc_u) - tf.math.log(rc_l)) * s[..., 4])
        return tf.stack([rom, rp, cp, h, rc], axis=-1)
    outputs = layers.Lambda(map_params, name='param_mapped')(raw)
    model = Model(inputs=inputs, outputs=outputs, name='parameter_nn')
    return model
def build_parameter_nn_multihead():
    # 电学输入：SOC、Tb、I
    elec_in = layers.Input(shape=(3,), name='elec_inputs')
    # 热学输入：Tb、Tin、Tamb、flow、speed
    therm_in1 = layers.Input(shape=(1,), name='therm_inputs1')
    therm_in2 = layers.Input(shape=(1,), name='therm_inputs2')
    # 头1：电学参数（小网络 + L2 正则）
    x1 = layers.Dense(32, activation='relu',
                  kernel_regularizer=regularizers.l2(1e-4))(elec_in)
    x1 = layers.Dense(16, activation='relu',
                  kernel_regularizer=regularizers.l2(1e-4))(x1)
    raw_elec = layers.Dense(3, activation=None, name='raw_elec')(x1)  # Rom, Rp, Cp

    # 头2：热学参数（小网络 + L2 正则）,输出换热系数
    x2 = layers.Dense(16, activation='relu',
                  kernel_regularizer=regularizers.l2(1e-4))(therm_in1)
    x2 = layers.Dense(16, activation='relu',
                  kernel_regularizer=regularizers.l2(1e-4))(x2)
    raw_therm1 = layers.Dense(1, activation=None, name='raw_therm1')(x2)

    # 头3：热学参数（小网络 + L2 正则），输出热阻
    x3 = layers.Dense(16, activation='relu',
                  kernel_regularizer=regularizers.l2(1e-4))(therm_in2)
    x3 = layers.Dense(16, activation='relu',
                  kernel_regularizer=regularizers.l2(1e-4))(x3)
    raw_therm2 = layers.Dense(1, activation=None, name='raw_therm2')(x3)

    # 映射到物理范围
    s_e = tf.keras.activations.sigmoid(raw_elec)
    s_t1 = tf.keras.activations.sigmoid(raw_therm1)
    s_t2 = tf.keras.activations.sigmoid(raw_therm2)
    # 建议范围（可按你系统调整）
    rom_l, rom_u = 1e-4, 2     # Rom
    rp_l,  rp_u  = 1e-4, 2.0      # Rp
    cp_l,  cp_u  = 1e-4, 1e5      # Cp（跨度大，用对数映射）
    h_l,   h_u   = 0.1, 100.0     # h
    rc_l,  rc_u  = 1e-4, 100.0    # Rc（跨度大，用对数映射）

    # 线性映射
    rom = rom_l + (rom_u - rom_l) * s_e[..., 0]
    rp  = rp_l  + (rp_u  - rp_l)  * s_e[..., 1]
    h   = h_l   + (h_u   - h_l)   * s_t1[..., 0]

    # 对数映射
    log_cp_l = tf.math.log(cp_l); log_cp_u = tf.math.log(cp_u)
    # cp = tf.exp(log_cp_l + (log_cp_u - log_cp_l) * s_e[..., 2])
    cp = ops.exp(log_cp_l + (log_cp_u - log_cp_l) * s_e[..., 2:3])

    log_rc_l = tf.math.log(rc_l); log_rc_u = tf.math.log(rc_u)
    rc = ops.exp(log_rc_l + (log_rc_u - log_rc_l) * s_t2[..., 0])

    params = tf.concat([rom[..., None], rp[..., None], cp[..., None], h[..., None], rc[..., None]], axis=-1)
    model = Model(inputs=[elec_in, therm_in1, therm_in2], outputs=params, name='parameter_nn_multihead')
    return model






# --- 构造神经网络 ---
# 输入是 [SOC, Temp, Current]
#重新构建神经网络
nn_model = build_parameter_nn_multihead()
#读取预训练好的网络继续训练
#nn_model =  load_model(r'C:\Users\brucet\Desktop\taomodel\physics_informed_battery_model_sequential2.h5', compile=False)
optimizer = optimizers.Adam(learning_rate=0.0001)
#nn_model.summary()
# --- 准备常数 ---
# 物理模型中的所有查表MAP

constants = {
     'S': 2.0, 'mass': 500.0, 'Vc': 0.02,
    'specific_heat': 1200.0, 'batt_cap': 138, 'dt': 1.0, 'num_cell': 204.0,
    'soc_keys': tf.constant([0, 3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 97, 100], dtype=tf.float32),
    'temp_discharge_keys': tf.constant([-20, 25, 45], dtype=tf.float32),'temp_charge_keys': tf.constant([-20, 25, 45], dtype=tf.float32),
    'ocv_discharge': tf.constant(np.array([
        [3.228, 3.226, 3.136], [3.361, 3.354, 3.306], [3.449, 3.439, 3.419], [3.486, 3.475, 3.468],
        [3.537, 3.519, 3.509], [3.581, 3.562, 3.553], [3.616, 3.593, 3.585], [3.635, 3.625, 3.617],
        [3.652, 3.646, 3.645], [3.672, 3.665, 3.664], [3.698, 3.690, 3.688], [3.735, 3.726, 3.722],
        [3.786, 3.775, 3.769], [3.852, 3.842, 3.833], [3.929, 3.921, 3.910], [3.990, 3.981, 3.980],
        [4.050, 4.043, 4.031], [4.109, 4.104, 4.093], [4.160, 4.157, 4.147], [4.201, 4.200, 4.190],
        [4.255, 4.255, 4.240], [4.296, 4.294, 4.280], [4.357, 4.352, 4.340]
    ]), dtype=tf.float32),
    'ocv_charge': tf.constant(np.array([
        [3.228, 3.226, 3.136], [3.354, 3.363, 3.312], [3.438, 3.454, 3.429], [3.474, 3.489, 3.479],
        [3.520, 3.538, 3.525], [3.563, 3.572, 3.583], [3.595, 3.618, 3.610], [3.625, 3.638, 3.635],
        [3.645, 3.655, 3.653], [3.665, 3.675, 3.673], [3.690, 3.701, 3.698], [3.726, 3.738, 3.733],
        [3.775, 3.788, 3.781], [3.844, 3.852, 3.841], [3.922, 3.930, 3.918], [3.982, 3.990, 3.978],
        [4.044, 4.051, 4.038], [4.105, 4.111, 4.098], [4.157, 4.162, 4.151], [4.199, 4.204, 4.192],
        [4.253, 4.257, 4.236], [4.295, 4.295, 4.278], [4.357, 4.352, 4.340]
    ]), dtype=tf.float32)
}

@tf.function
def train_on_single_sequence(sequence_tensor):
    """
    对一个完整的序列（来自一个CSV文件）进行前向模拟和梯度更新。
    Args:
        sequence_tensor: 一个形状为 (T, F) 的张量，T是序列长度，F是特征数。
    """
    sequence_length = tf.shape(sequence_tensor)[0]
    with tf.GradientTape() as tape:
        # --- a. 初始化状态 ---
        # 从真实数据的第一步获取初始状态
        initial_real_data = sequence_tensor[0]
        # 假设电流定义：放电为负，充电为正
        # ECM模型 V = OCV + I*R (因为I为负时，电压降低)
        # SOC' = I / (3600 * Cap) (因为I为正时，SOC增加)
        Ut0 = initial_real_data[1]      # PackVoltage
        Tb0 = initial_real_data[2]      # PackT
        Tout0 = initial_real_data[5]    # PackOutlet
        SOC0 = initial_real_data[3]     # PackSOC
        OCV = OCV_cell(constants, SOC0, Tb0, initial_real_data[0]) * 204
        Up0 = tf.constant(0.0, dtype=tf.float32) # 隐状态
        # 初始模拟状态元组
        sim_state = (Ut0, Tb0, Tout0, SOC0, Up0)

        # 用于存储每一步模拟结果的容器
        predicted_voltages = tf.TensorArray(tf.float32, size=sequence_length - 1)
        predicted_temps = tf.TensorArray(tf.float32, size=sequence_length - 1)
        predicted_Toutlet = tf.TensorArray(tf.float32, size=sequence_length - 1)
        predicted_SOC = tf.TensorArray(tf.float32, size=sequence_length - 1)
        # --- b. 循环模拟整个序列 ---
        for t in tf.range(sequence_length - 1):
            # 获取当前模拟状态
            _, current_Tb, _, current_SOC, _ = sim_state
            # 获取当前和未来的真实电流，当前的真实入水口温度，流量，车速作为驱动
            current_I = sequence_tensor[t, 0]
            future_I = sequence_tensor[t + 1, 0]
            control_Tin = sequence_tensor[t, 4]
            control_flow = sequence_tensor[t, 6]
            vehspeed = sequence_tensor[t, 7]
            current_Tamb = sequence_tensor[t, 8]

            drivers = (future_I, control_Tin, control_flow, current_Tamb)
            # 神经网络输入: [SOC, Temp, Current]
            #nn_input = tf.stack([current_SOC, current_Tb, current_I, control_flow, vehspeed])
            #nn_input = tf.expand_dims(nn_input, 0)  # 增加batch维度
            # 神经网络预测内阻参数
            #params_pred = nn_model(nn_input)[0]  # [R_o, R_p, C_p, h, Rc]
            elec_input = tf.stack([current_SOC, current_Tb, current_I])[None, ...]  # [1,3]
            therm_input1 = tf.stack([vehspeed])[None, ...]  # [1,5]
            therm_input2 = tf.stack([control_flow])[None, ...]  # [1,5]
            params_pred = nn_model([elec_input, therm_input1, therm_input2])[0]  # [5]
            # 运行可微分物理模型，预测下一步状态
            sim_state = differentiable_battery_step(sim_state, drivers, params_pred, constants)
            # 记录预测的电压和温度
            predicted_voltages = predicted_voltages.write(t, sim_state[0])  # Ut
            predicted_temps = predicted_temps.write(t, sim_state[1])  # Tb
            predicted_Toutlet = predicted_Toutlet.write(t, sim_state[2])  # Tb
            predicted_SOC = predicted_SOC.write(t, sim_state[3])  # Tb
        # --- c. 计算总损失 ---
        # 将TensorArray转换为普通Tensor
        pred_V = predicted_voltages.stack()
        pred_T = predicted_temps.stack()
        pred_Tout = predicted_Toutlet.stack()
        pred_SOC = predicted_SOC.stack()
        # 获取真实值（从时间步1开始，因为我们预测的是t+1的状态）
        true_V = sequence_tensor[1:, 1]
        true_T = sequence_tensor[1:, 2]
        true_Tout = sequence_tensor[1:, 5]
        true_SOC = sequence_tensor[1:, 3]
        # 计算均方误差
        loss_V = tf.reduce_mean(tf.square(pred_V - true_V))
        loss_T = tf.reduce_mean(tf.square(pred_T - true_T))
        loss_Tout = tf.reduce_mean(tf.square(pred_Tout - true_Tout))
        loss_SOC = tf.reduce_mean(tf.square(pred_SOC - true_SOC))
        # 加权总损失 (电压和温度的量级不同，需要加权)
        total_loss = loss_V + loss_T + loss_Tout + loss_SOC
    # --- d. 执行一次梯度更新 ---
    gradients = tape.gradient(total_loss, nn_model.trainable_variables)
    # 梯度裁剪，防止梯度爆炸，这对于RNN和长序列模拟很有帮助
    gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
    optimizer.apply_gradients(zip(gradients, nn_model.trainable_variables))
    return total_loss, loss_V, loss_T, loss_Tout, loss_SOC, true_V, pred_V


# --- 2. 主训练循环 ---
EPOCHS = 2
# folder_path = r'D:\taomodel\taotrain\train'
folder_path = r'/home/work/liuchunxiao/workspace/taomodel/taomodel/taotrain/train'

debug_data_to_plot = None
stop_training_flag = False
csv_file_pattern = os.path.join(folder_path, '*.csv')
csv_files = glob.glob(csv_file_pattern)
for epoch in range(EPOCHS):
    print(f"\n{'=' * 20} Epoch {epoch + 1}/{EPOCHS} {'=' * 20}")
    # 在每个epoch开始时打乱文件列表，这非常重要！
    random.shuffle(csv_files)
    # 使用tqdm创建进度条
    pbar = tqdm(csv_files, desc=f"Epoch {epoch + 1} 训练中")
    epoch_total_loss = 0
    epoch_v_loss = 0
    epoch_t_loss = 0
    epoch_tout_loss = 0
    epoch_SOC_loss = 0
    for i, file_path in enumerate(pbar):
        try:
            # --- a. 读取并预处理单个文件 ---
            real_data_df = read_battery_data(file_path)

            if real_data_df.empty:
                continue
            # 选择训练所需的列，顺序必须正确
            required_columns = ['PackCurrent', 'PackVoltage', 'PackT', 'PackSOC', 'PackInlet', 'PackOutlet', 'PumpFlow', 'VehicleSpeed', 'Tamb']
            processed_df = real_data_df[required_columns]
            # 处理缺失值
            processed_df = processed_df.ffill().bfill()
            if processed_df.isnull().values.any():
                print(f"文件 {os.path.basename(file_path)} 填充后仍有缺失值，已跳过。")
                continue
            # --- b. 将DataFrame转换为Tensor ---
            sequence_tensor = tf.convert_to_tensor(processed_df.to_numpy(dtype=np.float32))
            # --- c. 对这个文件（序列）进行训练 ---
            total_loss, loss_V, loss_T, loss_Tout, loss_SOC, true_V, pred_V  = train_on_single_sequence(sequence_tensor)
            # 累加损失用于显示

            if loss_V.numpy() > 50000.0:
                print("\n" + "=" * 50)
                print(f"!!!!!! 触发高电压损失条件: loss_V > 200.0 !!!!!!")
                print(f"文件名: {os.path.basename(file_path)}")
                print(
                    f"当前损失值: loss_T = {loss_T.numpy():.4f}, loss_V = {loss_V.numpy():.4f}, loss_Tout = {loss_Tout.numpy():.4f}, loss_SOC = {loss_SOC.numpy():.4f},")
                # 将需要的数据保存到我们之前定义的容器中
                debug_data_to_plot = {
                    "true_v": true_V.numpy(),
                    "pred_v": pred_V.numpy(),
                    "file_name": os.path.basename(file_path)
                }
                # 设置标志位，停止后续的训练
                stop_training_flag = True
                break  # 跳出当前的文件循环 (for i, file_path in enumerate(pbar):)
            epoch_total_loss += total_loss.numpy()
            epoch_v_loss += loss_V.numpy()
            epoch_t_loss += loss_T.numpy()
            epoch_tout_loss += loss_Tout.numpy()
            epoch_SOC_loss += loss_SOC.numpy()
            # 更新进度条显示平均损失
            pbar.set_postfix({
                'Avg Loss': f'{epoch_total_loss / (i + 1):.4f}',
                'Avg V_Loss': f'{epoch_v_loss / (i + 1):.4f}',
                'Avg T_Loss': f'{epoch_t_loss / (i + 1):.4f}',
                'Avg Tout_Loss': f'{epoch_tout_loss / (i + 1):.4f}',
                'Avg SOC_Loss': f'{epoch_SOC_loss / (i + 1):.4f}',
            })
        except Exception as e:
            print(f"\n处理文件 {os.path.basename(file_path)} 时发生错误: {e}")
# --- 训练结束后 ---
if not stop_training_flag:
    print("\n训练完成！")
    nn_model.save('/home/work/liuchunxiao/workspace/taomodel/taomodel/physics_informed_battery_model_sequential4.h5')
    print("模型已保存。")
else:
    print("\n训练因触发条件而提前终止。")

# 检查调试容器中是否有数据，如果有，则绘图

if debug_data_to_plot:
    print("\n正在处理触发高损失条件的数据...")
    # --- 1. 从容器中取出数据 ---
    true_v = debug_data_to_plot["true_v"]
    pred_v = debug_data_to_plot["pred_v"]
    original_csv_name = debug_data_to_plot["file_name"]
    # --- 2. 将数据保存到 Excel 文件 ---
    print("正在将数据保存到 Excel...")
    # a. 将两条数据放入一个字典，键将成为 Excel 的列名
    data_for_excel = {
        'True_Voltage': true_v,
        'Predicted_Voltage': pred_v
    }
    # b. 使用字典创建 Pandas DataFrame
    df = pd.DataFrame(data_for_excel)
    # c. 定义输出目录和文件名
    output_dir = 'plot_results'
    os.makedirs(output_dir, exist_ok=True)  # 确保目录存在
    base_name = os.path.splitext(original_csv_name)[0]
    excel_save_path = os.path.join(output_dir, f"{base_name}_data.xlsx")
    # d. 保存 DataFrame 到 Excel 文件
    # index=False 表示不将 DataFrame 的行索引（0, 1, 2...）写入文件
    df.to_excel(excel_save_path, index=False)
    print(f"数据已保存到: {excel_save_path}")

# 保存模型




