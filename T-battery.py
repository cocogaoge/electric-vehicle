import pandas as pd
import numpy as np
from scipy.interpolate import RegularGridInterpolator
def battery(battery_data, Rom, Rp, Cp):
    #电流充电为正，放电为负
    last_battery_data = battery_data.iloc[-1]#列表取最后一行
    current0 = last_battery_data['PackCurrent']
    Ut0 = last_battery_data['PackVoltage']
    Tb0 = last_battery_data['PackT']
    Tin0 = last_battery_data['PackInlet']
    Tout0 = last_battery_data['PackOutlet']
    SOC0 = last_battery_data['PackSOC']
    Rom0 = last_battery_data['PackRom']
    Rp0 = last_battery_data['PackRp']
    Cp0 = last_battery_data['PackCp']
    current = last_battery_data['future_Current']
    Up0 = current0 * Rp0

    #参数设置
    h = 20
    S = 2
    Rc = 0.1
    Tamb = 25
    mass = 500
    Vc = 0.02 #液冷板中的乙二醇溶液体积
    specific_heat = 1200
    mdot = 0.5 #kg/s
    batt_cap = 20 #Ah
    dt = 1
    # SOC 充电ocv 放电ocv
    soc_ocv_lemans_5_soc_key = np.array(
        [0, 3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 97, 100, ])
    soc_ocv_lemans_5_discharge_temp_key = np.array([-20, 25, 45])
    soc_ocv_lemans_5_charge_temp_key = np.array([25, 45])
    soc_ocv_lemans_5_discharge = np.array([
        [3.228, 3.226, 3.136], [3.361, 3.354, 3.306], [3.449, 3.439, 3.419], [3.486, 3.475, 3.468],
        [3.537, 3.519, 3.509], [3.581, 3.562, 3.553], [3.616, 3.593, 3.585], [3.635, 3.625, 3.617],
        [3.652, 3.646, 3.645], [3.672, 3.665, 3.664], [3.698, 3.690, 3.688], [3.735, 3.726, 3.722],
        [3.786, 3.775, 3.769], [3.852, 3.842, 3.833], [3.929, 3.921, 3.910], [3.990, 3.981, 3.980],
        [4.050, 4.043, 4.031], [4.109, 4.104, 4.093], [4.160, 4.157, 4.147], [4.201, 4.200, 4.190],
        [4.255, 4.255, 4.240], [4.296, 4.294, 4.280], [4.357, 4.352, 4.340]

    ])
    soc_ocv_lemans_5_charge = np.array([
        [3.226, 3.226], [3.363, 3.354], [3.454, 3.439], [3.489, 3.475],
        [3.538, 3.519], [3.583, 3.562], [3.618, 3.593], [3.638, 3.625],
        [3.655, 3.646], [3.675, 3.665], [3.701, 3.690], [3.738, 3.726],
        [3.788, 3.775], [3.852, 3.842], [3.930, 3.921], [3.990, 3.981],
        [4.051, 4.043], [4.111, 4.104], [4.162, 4.157], [4.204, 4.200],
        [4.257, 4.255], [4.295, 4.294], [4.352, 4.352]

    ])
    points1 = (soc_ocv_lemans_5_soc_key, soc_ocv_lemans_5_discharge_temp_key)
    points2 = (soc_ocv_lemans_5_soc_key, soc_ocv_lemans_5_charge_temp_key)

    # 创建放电插值函数
    interp_discharge = RegularGridInterpolator(points1, soc_ocv_lemans_5_discharge)
    # 创建充电插值函数
    interp_charge = RegularGridInterpolator(points2, soc_ocv_lemans_5_charge)
    # 电池数量
    num_cell = 204
    if current >= 0:
        OCV = interp_charge([SOC0, Tb0])*num_cell
    else:
        OCV = interp_discharge([SOC0, Tb0])*num_cell


    # 上一时刻状态推导出下一时刻状态
    SOC = SOC0 + current * dt / (3600 * batt_cap)
    Up =  (current / Cp - Up0 / (Rp * Cp)) * dt + Up0
    Ut = OCV + Up + current * Rom
    # Heat generation
    heat = 1.2 * abs(current * (OCV - Ut))
    # Heat dissipation
    disheat = h * S * (Tb0 - Tamb) + (Tb0 - Tout0) / Rc
    # Tb update
    Tb = (heat - disheat) * dt / (mass * specific_heat) + Tb0
    # Tc update
    Tc = (mdot * 3400 * (Tin0 - Tout0) + (Tb - Tout0) / Rc + 0.27 * 50 * (Tamb - Tout0) ) / (Vc * 1050 * 3400) * dt + Tout0

    new_row_data = {'PackCurrent': current, 'PackVoltage': Ut, 'PackT': Tb, 'PackOutlet': Tc, 'PackSOC': SOC, 'PackRom': Rom, 'PackRp': Rp, 'PackCp': Cp}
    battery_data.loc[len(battery_data)] = new_row_data#更新数据添加到最后一行
    return battery_data

