'''
Author: Fujin Wang
Date: 2024/5/28
Github: https://github.com/wang-fujin

Description:
该代码用于读取同济大学公开的电池数据集，方便后续预处理和分析。
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Battery:
    def __init__(self,path):
        pkl = pd.read_pickle(path)
        self.battery_id = path.split('/')[-1].split('.')[0]
        self.data = pkl[path.split('/')[-1].split('.')[0]]
        self.cycle_data = self.data['data']
        self.rul = self.data['rul']
        self.dq = self.data['dq']
        self.cycle_life = len(self.cycle_data)
        self.variables = ['Status', 'Cycle number', 'Current (mA)', 'Voltage (V)','Capacity (mAh)', 'Time (s)']

        print('-'*40,f' Battery {self.battery_id} ','-'*40)
        print('电池寿命：',self.cycle_life)
        print('变量名：',self.variables)
        #print('RUL:',self.rul)
        #print('dQ:',self.dq)
        print('-'*100)


    def _check(self,cycle=None,variable=None):
        '''
        检查输入的cycle和variable是否合法
        :param cycle: int: 循环次数
        :param variable: str: 变量名
        :return: bool: 是否合法
        '''
        if cycle is not None:
            if cycle > self.cycle_life or cycle < 1:
                raise ValueError('cycle should be in [1,{}]'.format(self.cycle_life))
        if variable is not None:
            if variable not in self.variables:
                raise ValueError('variable should be in {}'.format(self.variables))
        return True

    def print_variable_name(self):
        print('0: Status, '
              '1: Cycle number, '
              '2: Current (mA), '
              '3: Voltage (V), '
              '4: Capacity (mAh), '
              '5: Time (s)')

    def get_cycle(self,cycle):
        '''
        获取第cycle次循环的数据
        :param cycle: int: 循环次数
        :return: DataFrame: 第cycle次循环的数据, columns:['Status', 'Cycle number', 'Current (mA)', 'Voltage (V)','Capacity (mAh)', 'Time (s)']
        '''
        assert cycle <= self.cycle_life
        cycle_df = self.cycle_data[cycle]
        return cycle_df

    def get_degradation_trajectory(self):
        '''
        获取电池的退化轨迹
        :return:
        '''
        capacity = []
        for cycle in range(1,self.cycle_life+1):
            df = self.get_cycle(cycle)
            c = df['Capacity (mAh)'].max()
            capacity.append(c)
        return capacity


    def get_value(self,cycle,variable):
        '''
        获取第cycle次循环的variable变量的值
        :param cycle: int: 循环次数
        :param variable: str: 变量名
        :return: series: 第cycle次循环的variable变量的值
        '''
        self._check(cycle,variable)
        cycle_df = self.get_cycle(cycle)
        value = cycle_df[variable]
        return value

    def get_CCCV_stage(self,cycle):
        '''
        获取第cycle次循环的CCCV阶段的数据
        :param cycle: int: 循环次数
        :return: DataFrame: 第cycle次循环的CCCV阶段的数据, columns:['Status', 'Cycle number', 'Current (mA)', 'Voltage (V)','Capacity (mAh)', 'Time (s)']
        '''
        self._check(cycle)
        cycle_df = self.get_cycle(cycle)
        CCCV_df = cycle_df.loc[cycle_df['Status']=='Constant current-constant voltage charge',:]
        return CCCV_df

    def get_CC_stage(self,cycle,voltage_range=None):
        '''
        获取第cycle次循环的CC阶段的数据
        :param cycle: int: 循环次数
        :param voltage_range: list: 电压范围,[3.4,3.595]
        :return: DataFrame: 第cycle次循环的CC阶段的数据, columns:['Status', 'Cycle number', 'Current (mA)', 'Voltage (V)','Capacity (mAh)', 'Time (s)']
        '''
        self._check(cycle)
        CCCV_df = self.get_CCCV_stage(cycle)
        CC_CV_split_point = CCCV_df.loc[CCCV_df['Voltage (V)'] >= 3.595, :].index[0]
        CC_df = CCCV_df.loc[:CC_CV_split_point,:]  # CC阶段的数据

        if voltage_range == None:
            return CC_df
        else:
            CC_df = CC_df.loc[(CC_df['Voltage (V)']>=voltage_range[0]) & (CC_df['Voltage (V)']<=voltage_range[1]),:]
            return CC_df

    def get_CV_stage(self,cycle,current_range=None):
        '''
        获取第cycle次循环的CV阶段的数据
        :param cycle: int: 循环次数
        :param current_range: list: 电流范围,[1100,1000]
        :return: DataFrame: 第cycle次循环的CV阶段的数据, columns:['Status', 'Cycle number', 'Current (mA)', 'Voltage (V)','Capacity (mAh)', 'Time (s)']
        '''
        self._check(cycle)
        CCCV_df = self.get_CCCV_stage(cycle)
        CC_CV_split_point = CCCV_df.loc[CCCV_df['Voltage (V)'] >= 3.595, :].index[0]
        CV_df = CCCV_df.loc[CC_CV_split_point:,:]
        if current_range == None:
            return CV_df
        else:
            CV_df = CV_df.loc[(CV_df['Current (mA)']>=current_range[1]) & (CV_df['Current (mA)']<=current_range[0]),:]
            return CV_df

    def get_one_cycle_description(self,cycle):
        '''
        获取第cycle次循环的描述信息
        :param cycle: int: 循环次数
        :return: dict: 第cycle次循环的描述信息
        '''
        self._check(cycle)
        cycle_df = self.get_cycle(cycle)
        description = {}
        # state信息
        state = []
        for s in cycle_df['Status']:
            if s in state:
                continue
            else:
                state.append(s)
        description['Status'] = state

        # cycle number
        description['Cycle number'] = cycle_df['Cycle number'].iloc[0]

        # current
        currents = []
        for s in description['Status']:
            currents.append(cycle_df[cycle_df['Status']==s]['Current (mA)'].max())
        description['Current (mA)'] = currents

        return description

    def plot_one_cycle(self,cycle):
        '''
        绘制第cycle次循环的变量变化曲线
        :param cycle: int: 循环次数
        :return: None
        '''
        self._check(cycle)
        cycle_df = self.get_cycle(cycle)
        time = cycle_df['Time (s)']
        current = cycle_df['Current (mA)']
        voltage = cycle_df['Voltage (V)']
        capacity = cycle_df['Capacity (mAh)']
        fig, ax = plt.subplots(3, 1, figsize=(5, 5))
        ax[0].plot(time, current, color='b', linewidth=2)
        ax[0].set_ylabel('Current (mA)')
        ax[1].plot(time, voltage, color='r', linewidth=2)
        ax[1].set_ylabel('Voltage (V)')
        ax[2].plot(time, capacity, color='g', linewidth=2)
        ax[2].set_xlabel('Time (s)')
        ax[2].set_ylabel('Capacity (mAh)')
        plt.tight_layout()
        plt.show()

    def plot_one_cycle_CCCV(self,cycle):
        '''
        绘制第cycle次循环的CCCV阶段的曲线
        :param cycle: int: 循环次数
        :return: None
        '''
        self._check(cycle)
        # 在3行1列的图中分别画电流、电压、容量,并在每个子图中画出CC_CV_split_time的垂线
        fig, ax = plt.subplots(3, 1, figsize=(5, 5), dpi=200)

        CC_df = self.get_CC_stage(cycle)
        CV_df = self.get_CV_stage(cycle)

        ax[0].plot(CC_df['Time (s)'], CC_df['Current (mA)'], color='b', linewidth=2)
        ax[0].plot(CV_df['Time (s)'], CV_df['Current (mA)'], color='r', linewidth=2)
        ax[0].axvline(x=CC_df.iloc[-1,-1], color='k', linestyle='--')
        ax[0].set_ylabel('Current (mA)')

        ax[1].plot(CC_df['Time (s)'], CC_df['Voltage (V)'], color='b', linewidth=2)
        ax[1].plot(CV_df['Time (s)'], CV_df['Voltage (V)'], color='r', linewidth=2)
        ax[1].axvline(x=CC_df.iloc[-1,-1], color='k', linestyle='--')
        ax[1].set_ylabel('Voltage (V)')

        ax[2].plot(CC_df['Time (s)'], CC_df['Capacity (mAh)'], color='b', linewidth=2)
        ax[2].plot(CV_df['Time (s)'], CV_df['Capacity (mAh)'], color='r', linewidth=2)
        ax[2].axvline(x=CC_df.iloc[-1,-1], color='k', linestyle='--')
        ax[2].set_xlabel('Time (s)')
        ax[2].set_ylabel('Capacity (mAh)')
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    # 一个简单的例子
    path = '../our_data/1-1.pkl'
    battery = Battery(path)
    capacity = battery.get_degradation_trajectory()
    plt.plot(capacity)
    plt.show()

    battery.plot_one_cycle(1500)
    battery.plot_one_cycle_CCCV(1500)








