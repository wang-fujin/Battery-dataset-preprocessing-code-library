'''
Author: Fujin Wang
Date: 2025/10
Github: https://github.com/wang-fujin
Description: 
该代码用于读取RWTH公开的电池数据集，方便后续预处理和分析。
数据集链接：https://publications.rwth-aachen.de/record/818642
该数据集不是满充满放
'''

import os
import pandas as pd
import numpy as np
from pathlib import Path
import zipfile
from tqdm import tqdm
import matplotlib.pyplot as plt


class RWTHBatteryClass:
    def __init__(self, data_root='../../RWTH数据集'):
        # 该数据集有多层压缩，需逐步解压
        raw_file = Path(data_root)

        subdir = raw_file / 'RWTH-2021-04545_818642'
        if not (subdir / 'Rawdata.zip').exists():
            with zipfile.ZipFile(raw_file, 'r') as zip_ref:
                zip_ref.extractall(raw_file.parent)


        with zipfile.ZipFile(subdir / 'Rawdata.zip', 'r') as zip_ref:
            files = zip_ref.namelist()
            for file in files:
                if "BOL" not in file and not (subdir / file).exists():
                    zip_ref.extract(file, subdir)

        data_dir = subdir / 'Rohdaten'
        files = list(data_dir.glob('*.zip'))
        files = tqdm(files, desc='Unzipping files')
        for file in files:
            if not (data_dir / f'{file.stem}.csv').exists():
                with zipfile.ZipFile(file, 'r') as zip_ref:
                    zip_ref.extractall(data_dir)

        self.cells = [f'{i:03}' for i in range(2, 50)]

        self.data_dir = data_dir


        # 原始CSV文件的列名是德语的，对应的英文如下
        self.german_to_english = {
            "Schritt": "Step",
            "Zustand": "State",
            "Zeit": "Time",
            "Programmdauer": "Program Duration",
            "Schrittdauer": "Step Duration",
            "Zyklus": "Cycle",
            "Zyklusebene": "Cycle Stage",
            "Prozedur": "Procedure",
            "Prozedurebene": "Procedure Stage",
            "AhAkku": "Battery Ah",
            "AhLad": "Charged Ah",
            "AhEla": "Discharged Ah",
            "AhStep": "Step Ah",
            "Energie": "Energy",
            "WhStep": "Step Wh",
            "Spannung": "Voltage",
            "Strom": "Current",
            "Temp1": "Temperature"
        }


    def get_one_battery(self, cell_id='049'):
        """
        读取一个电池的所有CSV文件，拼接成一个DataFrame，并返回。
        同时，把这个DataFrame保存为pkl文件，以便下次快速读取。
        :param cell_id: 电池ID，str类型，形如‘002’，‘010’，‘025’等
        :return: DataFrame
        """
        battery_file = self.data_dir / f'RWTH_{cell_id}-TBA_Zyk.pkl'

        if battery_file.exists():
            print(f'Battery {cell_id} has been processed before, loading from {battery_file}')
            return pd.read_pickle(battery_file)
        else:
            print(f'Processing cell {cell_id}')
            this_battery_files = self.data_dir.glob(f'*{cell_id}=ZYK*Zyk*.csv')
            df = pd.concat([pd.read_csv(f, skiprows=[1]) for f in this_battery_files])

            df.columns = [self.german_to_english[col] for col in df.columns]
            print('Dataframe shape:',df.shape)

            df = df.drop_duplicates('Time').sort_values('Time')
            df = df[find_time_anomalies(df['Program Duration'].values)]
            df = df.reset_index(drop=True)
            cycle_ends = find_cycle_ends(df['Current'].values)
            # 跳过第一个cycle，因为放电阶段不完整
            cycle_ends = df['Current'][cycle_ends].index[1:]

            desc = f'Processing each cycles of cell {cell_id}'
            df1 = df.iloc[cycle_ends[0]:].copy()
            for i in tqdm(range(1, len(cycle_ends)), desc=desc):
                df1.loc[cycle_ends[i-1]:cycle_ends[i], 'Cycle'] = i

            processed_df = df1[['State','Time', 'Step Duration', 'Cycle', 'Voltage',
                               'Current', 'Temperature', 'Charged Ah', 'Discharged Ah', 'Battery Ah', 'Step Ah'
                               ]]
            processed_df.to_pickle(battery_file)
            print(f"Saved DataFrame to {battery_file}")
            return processed_df


    def get_one_battery_one_cycle(self, cell_id, cycle):
        '''
        获取一个电池的一个cycle的数据，返回一个DataFrame
        :param cell_id: 电池ID，str类型，形如‘002’，‘010’，‘025’等
        :param cycle: int类型，循环次数
        :return: DataFrame
        '''
        if cell_id not in self.cells:
            raise ValueError(f'Cell {cell_id} not found')

        df = self.get_one_battery(cell_id)
        cycles = df['Cycle'].unique()
        if cycle not in cycles:
            raise ValueError(f'Cycle {cycle} not found in cell {cell_id}')

        cycle_df = df[df['Cycle'] == cycle].copy()

        cycle_df['Charged Ah'] = cycle_df['Charged Ah'] - cycle_df['Charged Ah'].iloc[0]
        cycle_df['Discharged Ah'] = cycle_df['Discharged Ah'] - cycle_df['Discharged Ah'].iloc[0]

        return cycle_df

    def get_one_battery_one_cycle_charge(self, cell_id, cycle, mode=None):
        '''
        获取一个电池的一个cycle的充电数据，返回一个DataFrame
        :param cell_id: 电池ID，str类型，形如‘002’，‘010’，‘025’等
        :param cycle: 周期编号，int类型
        :param mode: 'CC' 或 'CV' 或 'None'，充电模式，默认为None，表示不区分充电模式
        :return:
        '''
        one_cycle = self.get_one_battery_one_cycle(cell_id, cycle)
        charge_df = one_cycle[one_cycle['State'] == 'CHA'].copy()
        charge_df.reset_index(drop=True, inplace=True)

        if mode == 'CC':
            charge_df = charge_df[charge_df['Current'] > 4.0]
        elif mode == 'CV':
            charge_df = charge_df[charge_df['Current'] < 4.0]

        # 计算index差分，如果index差分大于1，则说明有异常数据，则删除
        index_diff = np.diff(charge_df.index)
        index_diff = np.where(index_diff > 1)[0]
        start_index = index_diff[-1]+1 if len(index_diff) > 0 else 0
        charge_df = charge_df.iloc[start_index:,:]

        return charge_df

    def get_one_battery_one_cycle_discharge(self, cell_id, cycle, mode=None):
        '''
        获取一个电池的一个cycle的放电数据，返回一个DataFrame
        :param cell_id: 电池ID，str类型，形如‘002’，‘010’，‘025’等
        :param cycle: 周期编号，int类型
        :param mode: 'CC' 或 'CV' 或 'None'，放电模式，默认为None，表示不区分放电模式
        [Note]:该数据集不是满充满放，存在CV放电
        :return:
       '''
        one_cycle = self.get_one_battery_one_cycle(cell_id, cycle)
        discharge_df = one_cycle[one_cycle['State'] == 'DCH'].copy()
        discharge_df.reset_index(drop=True, inplace=True)

        if mode == 'CC':
            discharge_df = discharge_df[discharge_df['Current'] < -4.0]
        elif mode == 'CV':
            discharge_df = discharge_df[discharge_df['Current'] > -4.0]

        # 计算index差分，如果index差分大于1，则说明有异常数据，则删除
        index_diff = np.diff(discharge_df.index)
        index_diff = np.where(index_diff > 1)[0]
        start_index = index_diff[-1]+1 if len(index_diff) > 0 else 0
        discharge_df = discharge_df.iloc[start_index:,:]

        return discharge_df




# 下面的函数参考了 https://github.com/microsoft/BatteryML/blob/main/batteryml/preprocess/preprocess_RWTH.py
def find_cycle_ends(current, lag=10, tolerance=0.1):
    is_cycle_end = np.zeros_like(current, dtype=np.bool8)
    enter_discharge_steps = 0
    for i in range(len(current)):
        I = current[i]  # noqa
        if i > 0 and i < len(current):
            # Process the non-smoothness
            if np.abs(current[i] - current[i-1]) > tolerance \
                    and np.abs(current[i] - current[i+1]) > tolerance:
                I = current[i+1]  # noqa
        if I < 0:  # discharge
            enter_discharge_steps += 1
        else:
            enter_discharge_steps = 0
        nms_size = 500
        if enter_discharge_steps == lag:
            t = i - lag + 1
            if t > nms_size and np.max(is_cycle_end[t-nms_size:t]) > 0:
                continue
            is_cycle_end[t] = True

    return is_cycle_end

def find_time_anomalies(time, tolerance=1e5):
    prev = time[0]
    result = np.ones_like(time, dtype=np.bool8)
    for i in range(1, len(time)):
        if time[i] - prev > tolerance:
            result[i] = False
        else:
            prev = time[i]
    return result

def remove_abnormal_cycle(Qd, eps=0.05, window=5):
    to_remove = np.zeros_like(Qd, dtype=np.bool8)
    for i in range(window, len(Qd)-window):
        prev = max(0, i - window)
        if np.abs(Qd[i] - np.median(Qd[prev:i])) > eps \
                and np.abs(Qd[i] - np.median(Qd[i:i+window])) > eps:
            to_remove[i] = True
    return to_remove

def calc_Q(I, t, is_charge):  # noqa
    Q = np.zeros_like(I)
    for i in range(1, len(I)):
        if is_charge and I[i] > 0:
            Q[i] = Q[i-1] + I[i] * (t[i] - t[i-1]) / 36e5
        elif not is_charge and I[i] < 0:
            Q[i] = Q[i-1] - I[i] * (t[i] - t[i-1]) / 36e5
        else:
            Q[i] = Q[i-1]
    return Q




if __name__ == '__main__':
    dataset = RWTHBatteryClass()
    cycle_df = dataset.get_one_battery_one_cycle_discharge('002', 1, 'CV')
    print(cycle_df.columns)
    print(cycle_df.shape)

    for col in cycle_df.columns:
        fig = plt.figure()
        plt.scatter(cycle_df.index,cycle_df[col])
        plt.title(col)
        plt.show()
        plt.close()

