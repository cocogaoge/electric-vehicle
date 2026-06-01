import os
import glob
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm
from pytorch_battery import differentiable_battery_step
from read_battery_data import read_battery_data # 确保你的读取函数可用

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------- 1. 物理常数配置 --------------------------
def get_constants(device):
    # 与TensorFlow版本一致的常数字典
    return {
        'S': 2.0, 'mass': 500.0, 'Vc': 0.02,
        'specific_heat': 1200.0, 'batt_cap': 138.0, 'dt': 1.0, 'num_cell': 204.0,
        'soc_keys': torch.tensor([0, 3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 97, 100], dtype=torch.float32, device=device),
        'temp_discharge_keys': torch.tensor([-20, 25, 45], dtype=torch.float32, device=device),
        'temp_charge_keys': torch.tensor([-20, 25, 45], dtype=torch.float32, device=device),
        'ocv_discharge': torch.tensor([
            [3.228, 3.226, 3.136], [3.361, 3.354, 3.306], [3.449, 3.439, 3.419], [3.486, 3.475, 3.468],
            [3.537, 3.519, 3.509], [3.581, 3.562, 3.553], [3.616, 3.593, 3.585], [3.635, 3.625, 3.617],
            [3.652, 3.646, 3.645], [3.672, 3.665, 3.664], [3.698, 3.690, 3.688], [3.735, 3.726, 3.722],
            [3.786, 3.775, 3.769], [3.852, 3.842, 3.833], [3.929, 3.921, 3.910], [3.990, 3.981, 3.980],
            [4.050, 4.043, 4.031], [4.109, 4.104, 4.093], [4.160, 4.157, 4.147], [4.201, 4.200, 4.190],
            [4.255, 4.255, 4.240], [4.296, 4.294, 4.280], [4.357, 4.352, 4.340]
        ], dtype=torch.float32, device=device),
        'ocv_charge': torch.tensor([
            [3.228, 3.226, 3.136], [3.354, 3.363, 3.312], [3.438, 3.454, 3.429], [3.474, 3.489, 3.479],
            [3.520, 3.538, 3.525], [3.563, 3.572, 3.583], [3.595, 3.618, 3.610], [3.625, 3.638, 3.635],
            [3.645, 3.655, 3.653], [3.665, 3.675, 3.673], [3.690, 3.701, 3.698], [3.726, 3.738, 3.733],
            [3.775, 3.788, 3.781], [3.844, 3.852, 3.841], [3.922, 3.930, 3.918], [3.982, 3.990, 3.978],
            [4.044, 4.051, 4.038], [4.105, 4.111, 4.098], [4.157, 4.162, 4.151], [4.199, 4.204, 4.192],
            [4.253, 4.257, 4.236], [4.295, 4.295, 4.278], [4.357, 4.352, 4.340]
        ], dtype=torch.float32, device=device)
    }

# -------------------------- 2. 模型定义 --------------------------
# A. 物理信息神经网络 (PINN) - 对应你的 TF 模型
class BatteryPINN(nn.Module):
    def __init__(self):
        super(BatteryPINN, self).__init__()
        
        # 归一化参数
        self.register_buffer('elec1_mean', torch.tensor([0., 10., 0.]))
        self.register_buffer('elec1_std', torch.tensor([100., 35., 200.]))
        self.register_buffer('elec2_mean', torch.tensor([0., 10.]))
        self.register_buffer('elec2_std', torch.tensor([100., 35.]))
        self.register_buffer('therm1_scale', torch.tensor(150.0))
        self.register_buffer('therm2_scale', torch.tensor(10.0))

        # 头1：电学参数 (Rom, Rp, Cp)
        self.head1 = nn.Sequential(
            nn.Linear(3, 32), nn.SiLU(),
            nn.Linear(32, 32), nn.SiLU(),
            nn.Dropout(0.2), nn.Linear(32, 3)
        )
        # 头2：电学参数 (OCV_bias)
        self.head2 = nn.Sequential(
            nn.Linear(2, 16), nn.SiLU(),
            nn.Linear(16, 8), nn.SiLU(),
            nn.Dropout(0.2), nn.Linear(8, 1)
        )
        # 头3：热力学参数 (h, hca)
        self.head3 = nn.Sequential(
            nn.Linear(1, 32), nn.SiLU(),
            nn.Linear(32, 16), nn.SiLU(),
            nn.Dropout(0.2), nn.Linear(16, 2)
        )
        # 头4：热力学参数 (Rc)
        self.head4 = nn.Sequential(
            nn.Linear(1, 16), nn.SiLU(),
            nn.Linear(16, 8), nn.SiLU(),
            nn.Dropout(0.2), nn.Linear(8, 1)
        )

    def forward(self, e1, e2, t1, t2):
        # 归一化
        e1 = (e1 - self.elec1_mean) / self.elec1_std
        e2 = (e2 - self.elec2_mean) / self.elec2_std
        t1 = t1 / self.therm1_scale
        t2 = t2 / self.therm2_scale

        out1 = torch.sigmoid(self.head1(e1))
        out2 = torch.sigmoid(self.head2(e2))
        out3 = torch.sigmoid(self.head3(t1))
        out4 = torch.sigmoid(self.head4(t2))

        # 映射到物理范围
        rom = 1e-4 + (2.0 - 1e-4) * out1[:, 0:1]
        rp = 1e-4 + (2.0 - 1e-4) * out1[:, 1:2]
        cp = torch.exp(torch.log(torch.tensor(1e-4)) + (torch.log(torch.tensor(1e5)) - torch.log(torch.tensor(1e-4))) * out1[:, 2:3])
        
        ocv_bias = -1.0 + 2.0 * out2[:, 0:1]
        
        h = 0.1 + (100.0 - 0.1) * out3[:, 0:1]
        hca = 0.1 + (1000.0 - 0.1) * out3[:, 1:2]
        
        rc = torch.exp(torch.log(torch.tensor(1e-4)) + (torch.log(torch.tensor(100.0)) - torch.log(torch.tensor(1e-4))) * out4[:, 0:1])

        return torch.cat([rom, rp, cp, ocv_bias, h, hca, rc], dim=-1)

# B. 纯 Transformer 模型 (直接映射)
class BatteryTransformer(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=64, num_layers=2, output_dim=4):
        super(BatteryTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(hidden_dim, output_dim) # 输出: V, T, Tout, SOC

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        embedded = self.embedding(x)
        out = self.transformer(embedded)
        preds = self.fc_out(out)
        return preds

# -------------------------- 3. 训练函数 --------------------------
def train_pinn_sequence(model, sequence_tensor, constants, optimizer):
    model.train()
    optimizer.zero_grad()
    
    seq_len = sequence_tensor.shape[0]
    initial_data = sequence_tensor[0]
    
    # 初始状态: Ut, Tb, Tout, SOC, Up
    sim_state = (initial_data[1], initial_data[2], initial_data[5], initial_data[3], torch.tensor(0.0, device=device))
    
    pred_V, pred_T, pred_Tout, pred_SOC = [], [], [], []
    
    for t in range(seq_len - 1):
        _, current_Tb, _, current_SOC, _ = sim_state
        current_I = sequence_tensor[t, 0]
        future_I = sequence_tensor[t + 1, 0]
        control_Tin = sequence_tensor[t, 4]
        control_flow = sequence_tensor[t, 6]
        vehspeed = sequence_tensor[t, 7]
        current_Tamb = sequence_tensor[t, 8]
        
        drivers = (future_I, control_Tin, control_flow, current_Tamb)
        
        # 准备模型输入
        e1 = torch.stack([current_SOC, current_Tb, current_I]).unsqueeze(0)
        e2 = torch.stack([current_SOC, current_Tb]).unsqueeze(0)
        t1 = vehspeed.view(1, 1)
        t2 = control_flow.view(1, 1)
        
        params_pred = model(e1, e2, t1, t2)[0]
        sim_state = differentiable_battery_step(sim_state, drivers, params_pred, constants)
        
        pred_V.append(sim_state[0])
        pred_T.append(sim_state[1])
        pred_Tout.append(sim_state[2])
        pred_SOC.append(sim_state[3])
        
    pred_V = torch.stack(pred_V)
    pred_T = torch.stack(pred_T)
    pred_Tout = torch.stack(pred_Tout)
    pred_SOC = torch.stack(pred_SOC)
    
    true_V = sequence_tensor[1:, 1]
    true_T = sequence_tensor[1:, 2]
    true_Tout = sequence_tensor[1:, 5]
    true_SOC = sequence_tensor[1:, 3]
    
    loss_V = torch.mean((pred_V - true_V)**2)
    loss_T = torch.mean((pred_T - true_T)**2)
    loss_Tout = torch.mean((pred_Tout - true_Tout)**2)
    loss_SOC = torch.mean((pred_SOC - true_SOC)**2)
    
    #
    total_loss = 0.1 * loss_V + 2 * loss_T + 3 * loss_Tout
    
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    return total_loss.item(), loss_V.item(), loss_T.item(), loss_Tout.item(), loss_SOC.item()

def train_transformer_sequence(model, sequence_tensor, optimizer):
    model.train()
    optimizer.zero_grad()
    
    # 增加batch维度 -> (1, seq_len, features)
    x = sequence_tensor[:-1].unsqueeze(0) 
    y_true = sequence_tensor[1:, [1, 2, 5, 3]].unsqueeze(0) # 预测下一步的 V, T, Tout, SOC
    
    preds = model(x)
    
    # 计算MSE Loss
    loss_V = torch.mean((preds[0, :, 0] - y_true[0, :, 0])**2)
    loss_T = torch.mean((preds[0, :, 1] - y_true[0, :, 1])**2)
    loss_Tout = torch.mean((preds[0, :, 2] - y_true[0, :, 2])**2)
    loss_SOC = torch.mean((preds[0, :, 3] - y_true[0, :, 3])**2)
    
    total_loss = 0.1 * loss_V + 2 * loss_T + 3 * loss_Tout + loss_SOC
    
    total_loss.backward()
    optimizer.step()
    
    return total_loss.item(), loss_V.item(), loss_T.item(), loss_Tout.item(), loss_SOC.item()

# -------------------------- 4. 主训练循环 --------------------------
if __name__ == "__main__":
    folder_path = r'./taotrain/trainlong' # 修改为你的训练数据路径
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
    
    pinn_model = BatteryPINN().to(device)
    transformer_model = BatteryTransformer().to(device)
    
    opt_pinn = optim.Adam(pinn_model.parameters(), lr=0.001)
    opt_trans = optim.Adam(transformer_model.parameters(), lr=0.001)
    
    constants = get_constants(device)
    EPOCHS = 10
    
    for epoch in range(EPOCHS):
        print(f"\n{'=' * 20} Epoch {epoch + 1}/{EPOCHS} {'=' * 20}")
        random.shuffle(csv_files)
        
        # 为了简洁，这里仅演示核心训练逻辑
        for file_path in tqdm(csv_files):
            df = read_battery_data(file_path)
            if df.empty: continue
            
            cols = ['PackCurrent', 'PackVoltage', 'PackT', 'PackSOC', 'PackInlet', 'PackOutlet', 'PumpFlow', 'VehicleSpeed', 'Tamb']
            # 直接把 DataFrame 转换为纯 Python 的嵌套列表 (list)，彻底绕开 NumPy 的底层 API
            data_list = df[cols].ffill().bfill().values.tolist()
            #    PyTorch 可以完美原生解析 Python list
            seq_tensor = torch.tensor(data_list, dtype=torch.float32, device=device)
            
            # 训练 PINN
            p_loss = train_pinn_sequence(pinn_model, seq_tensor, constants, opt_pinn)
            # 训练 Transformer
            t_loss = train_transformer_sequence(transformer_model, seq_tensor, opt_trans)
            
        print(f"PINN Loss: {p_loss[0]:.4f} | Transformer Loss: {t_loss[0]:.4f}")
        
    # 保存模型
    torch.save(pinn_model.state_dict(), 'pinn_model.pth')
    torch.save(transformer_model.state_dict(), 'transformer_model.pth')
    print("Models saved successfully.")