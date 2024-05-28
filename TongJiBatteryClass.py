'''
Author: Fujin Wang
Date: 2024.05
Github: https://github.com/wang-fujin

Description:
该代码用于读取同济大学公开的电池数据集，方便后续预处理和分析。
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pprint

class Battery:
    def __init__(self,path='../Dataset_1_NCA_battery/CY25-1_1-#1.csv'):
        self.path = path
        self.df = pd.read_csv(path)
        file_name = path.split('/')[-1]
        self.temperature = int(file_name[2:4])
        self.charge_c_rate = file_name.split('-')[1].split('_')[0]
        self.discharge_c_rate = file_name.split('-')[1].split('_')[1]
        self.battery_id = file_name.split('#')[-1].split('.')[0]
        self.cycle_index = self._get_cycle_index()
        self.cycle_life = len(self.cycle_index)
        print('-'*40,f' Battery #{self.battery_id} ','-'*40)
        print('电池寿命：',self.cycle_life)
        print('实验温度：',self.temperature)
        print('充电倍率：',self.charge_c_rate)
        print('放电倍率：',self.discharge_c_rate)
        print('变量名：',list(self.df.columns))
        print('-'*100)

    def _get_cycle_index(self):
        cycle_num = np.unique(self.df['cycle number'].values)
        return cycle_num

    def _check(self,cycle=None,variable=None):
        '''
        检查输入的cycle和variable是否合法
        :param cycle: int: 循环次数
        :param variable: str: 变量名
        :return: bool: 是否合法
        '''
        if cycle is not None:
            if cycle not in self.cycle_index:
                raise ValueError('cycle should be in [{},{}]'.format(int(self.cycle_index.min()),int(self.cycle_index.max())))
        if variable is not None:
            if variable not in self.df.columns:
                raise ValueError('variable should be in {}'.format(list(self.df.columns)))
        return True

    def get_cycle(self,cycle):
        '''
        获取第cycle次循环的数据
        :param cycle: int: 循环次数
        :return: DataFrame: 第cycle次循环的数据, columns:['time/s', 'control/V/mA', 'Ecell/V', '<I>/mA', 'Q discharge/mA.h', 'Q charge/mA.h', 'control/V', 'control/mA', 'cycle number']
        '''
        self._check(cycle=cycle)
        cycle_df = self.df[self.df['cycle number']==cycle]
        return cycle_df

    def get_degradation_trajectory(self):
        '''
        获取电池的容量退化轨迹
        :return:
        '''
        capacity = []
        for cycle in self.cycle_index:
            cycle_df = self.get_cycle(cycle)
            capacity.append(cycle_df['Q discharge/mA.h'].max())
        return capacity

    def get_value(self,cycle,variable):
        '''
        获取第cycle次循环的variable变量的值
        :param cycle: int: 循环次数
        :param variable: str: 变量名
        :return: series: 第cycle次循环的variable变量的值
        '''
        self._check(cycle=cycle,variable=variable)
        cycle_df = self.get_cycle(cycle)
        return cycle_df[variable].values

    def get_charge_stage(self,cycle):
        '''
        获取第cycle次循环的CCCV阶段的数据
        :param cycle: int: 循环次数
        :return: DataFrame: 第cycle次循环的CCCV阶段的数据, columns:['time/s', 'control/V/mA', 'Ecell/V', '<I>/mA', 'Q discharge/mA.h', 'Q charge/mA.h', 'control/V', 'control/mA', 'cycle number']
        '''
        self._check(cycle=cycle)
        cycle_df = self.get_cycle(cycle)
        charge_df = cycle_df[cycle_df['control/V/mA']>0]
        return charge_df

    def get_CC_stage(self,cycle,voltage_range=None):
        '''
        获取第cycle次循环的CC阶段的数据
        :param cycle: int: 循环次数
        :param voltage_range: list: 电压范围
        :return: DataFrame: 第cycle次循环的CC阶段的数据, columns:['time/s', 'control/V/mA', 'Ecell/V', '<I>/mA', 'Q discharge/mA.h', 'Q charge/mA.h', 'control/V', 'control/mA', 'cycle number']
        '''
        self._check(cycle=cycle)
        cycle_df = self.get_cycle(cycle)
        CC_df = cycle_df[cycle_df['control/mA']>0]

        if voltage_range is not None:
            CC_df = CC_df[CC_df['Ecell/V'].between(voltage_range[0],voltage_range[1])]
        return CC_df

    def get_CV_stage(self,cycle,current_range=None):
        '''
        获取第cycle次循环的CV阶段的数据
        :param cycle: int: 循环次数
        :param current_range: list: 电流范围
        :return: DataFrame: 第cycle次循环的CV阶段的数据, columns:['time/s', 'control/V/mA', 'Ecell/V', '<I>/mA', 'Q discharge/mA.h', 'Q charge/mA.h', 'control/V', 'control/mA', 'cycle number']
        '''
        self._check(cycle=cycle)
        cycle_df = self.get_cycle(cycle)
        CV_df = cycle_df[cycle_df['control/V']>0]

        if current_range is not None:
            CV_df = CV_df[CV_df['<I>/mA'].between(np.min(current_range),np.max(current_range))]
        return CV_df

    def plot_one_cycle_CCCV(self,cycle):
        '''
        绘制第cycle次循环的CCCV阶段的曲线
        :param cycle: int: 循环次数
        :return: None
        '''
        self._check(cycle=cycle)
        CC_df = self.get_CC_stage(cycle,voltage_range=[4.0,4.2])
        CV_df = self.get_CV_stage(cycle,current_range=[2000,1000])

        fig, ax = plt.subplots(3, 1, figsize=(5, 5), dpi=200)
        ax[0].plot(CC_df['time/s'].values, CC_df['Ecell/V'].values, color='b', linewidth=2)
        ax[0].plot(CV_df['time/s'].values, CV_df['Ecell/V'].values, color='r', linewidth=2)
        ax[0].set_ylabel('Voltage/V')
        ax[0].axvline(x=CC_df['time/s'].values[-1], color='g', linestyle='--')

        ax[1].plot(CC_df['time/s'].values, CC_df['<I>/mA'].values, color='b', linewidth=2)
        ax[1].plot(CV_df['time/s'].values, CV_df['<I>/mA'].values, color='r', linewidth=2)
        ax[1].axvline(x=CC_df['time/s'].values[-1], color='g', linestyle='--')
        ax[1].set_ylabel('Current/mA')

        ax[2].plot(CC_df['time/s'].values, CC_df['Q charge/mA.h'].values, color='b', linewidth=2)
        ax[2].plot(CV_df['time/s'].values, CV_df['Q charge/mA.h'].values, color='r', linewidth=2)
        ax[2].axvline(x=CC_df['time/s'].values[-1], color='g', linestyle='--')
        ax[2].set_ylabel('Charge Q/mA.h')
        ax[2].set_xlabel('Time/s')
        plt.tight_layout()
        plt.show()

    def plot_one_cycle(self,cycle):
        '''
        绘制第cycle次循环的变量的曲线
        :param cycle: int: 循环次数
        :return: None
        '''
        self._check(cycle=cycle)
        cycle_df = self.get_cycle(cycle)

        time = cycle_df['time/s'].values
        voltage = cycle_df['Ecell/V'].values
        current = cycle_df['<I>/mA'].values
        discharge_capacity = cycle_df['Q discharge/mA.h'].values
        charge_capacity = cycle_df['Q charge/mA.h'].values

        fig,ax = plt.subplots(4,1,figsize=(6,6),dpi=200)
        ax[0].plot(time,voltage,color='b', linewidth=2)
        ax[0].set_ylabel('Voltage/V')
        ax[1].plot(time,current,color='r', linewidth=2)
        ax[1].set_ylabel('Current/mA')
        ax[2].plot(time,discharge_capacity,color='g', linewidth=2)
        ax[2].set_ylabel('Q Discharge/mA.h')
        ax[3].plot(time,charge_capacity,color='y', linewidth=2)
        ax[3].set_ylabel('Q Charge/mA.h')
        ax[3].set_xlabel('Time/s')
        plt.suptitle(f'Battery {self.battery_id} Cycle {cycle}')
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    # 一个简单的例子
    path = '../Dataset_1_NCA_battery/CY45-05_1-#15.csv'
    battery = Battery(path=path)
    capacity = battery.get_degradation_trajectory()
    plt.plot(capacity)
    plt.show()

    # battery.plot_one_cycle(6)
    # battery.plot_one_cycle_CCCV(8)

