'''
Author: Fujin Wang
Date: 2024.05
Github: https://github.com/wang-fujin

Description:
该代码用于读取XJTU电池数据集，方便后续预处理和分析。
数据集链接：https://zenodo.org/records/11046967
如果您使用了该代码，请引用以下论文：
Wang, F., Zhai, Z., Zhao, Z. et al. Physics-informed neural network for lithium-ion battery degradation stable modeling and prognosis. Nat Commun 15, 4332 (2024). https://doi.org/10.1038/s41467-024-48779-z

This code is used to read the XJTU battery dataset to facilitate subsequent preprocessing and analysis.
The dataset link: https://zenodo.org/records/11046967
If you use this code, please cite our paper:
Wang, F., Zhai, Z., Zhao, Z. et al. Physics-informed neural network for lithium-ion battery degradation stable modeling and prognosis. Nat Commun 15, 4332 (2024). https://doi.org/ 10.1038/s41467-024-48779-z
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import interpolate
import os
import functools

def interpolate_resample(resample=True, num_points=128):
    '''
    插值重采样装饰器,如果resample为True，那么就进行插值重采样，点数为num_points,默认为128；
    否则就不进行重采样
    :param resample: bool: 是否进行重采样
    :param num_points: int: 重采样的点数
    :return:
    '''
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self,*args, **kwargs):
            data = func(self,*args, **kwargs)
            if resample:
                x = np.linspace(0, 1, data.shape[0])
                f1 = interpolate.interp1d(x, data, kind='linear')
                new_x = np.linspace(0, 1, num_points)
                data = f1(new_x)
            return data
        return wrapper
    return decorator


class Battery:
    def __init__(self,path):
        mat = loadmat(path)
        self.data = mat['data']
        self.battery_name = path.split('/')[-1].split('.')[0]
        self.summary = mat['summary']
        self.cycle_life = self.summary[0][0][8][0][0]
        self.description = self.summary[0][0][9][0]
        self.variable_name = ['system_time','relative_time_min','voltage_V','current_A','capacity_Ah','power_Wh','temperature_C','description']
        print(f'电池寿命：{self.cycle_life}, 电池描述：{self.description}')

    def print_variable_name(self):
        print('0:system_time, '
              '1:relative_time_min, '
              '2:voltage_V, '
              '3:current_A, '
              '4:capacity_Ah, '
              '5:power_Wh, '
              '6:temperature_C, '
              '7:description'
              )

    def get_descriptions(self):
        '''
        电池每个cycle都有一个描述，这个函数返回所有的描述种类
        :return:
        '''
        descriptions = []
        for i in range(self.data.shape[1]):
            description = self.data[0][i][7][0]
            if description not in descriptions:
                descriptions.append(description)
        return descriptions

    def get_one_cycle_description(self,cycle):
        '''
        获取某个cycle的描述
        :param cycle: int: 电池的循环次数
        :return:
        '''
        # 如果cycle大于电池的循环次数，或者小于1，那么就报错，并提示用户
        if cycle > self.data.shape[1] or cycle < 1:
            raise ValueError(f'cycle的值应该在[1,{self.data.shape[1]}]之内')
        description = self.data[0][cycle-1][7][0]
        return description

    def get_degradation_trajectory(self):
        '''
        获取电池的容量衰减轨迹。因为很多cycle没有完全放电，因此需要用[test capacity]的cycle来插值
        对于完全放电的前几批次数据，该函数与self.get_capacity()函数的结果是一样的
        :return:
        '''
        # 获取测量容量的cycle
        test_capacity_cycles = []
        for i in range(1,self.cycle_life+1):
            des = self.get_one_cycle_description(i)
            if 'test capacity' in des:
                test_capacity_cycles.append(i)
        # 获取测量容量的cycle的容量
        index = np.array(test_capacity_cycles)-1
        capacity = self.get_capacity()
        test_capacity = capacity[index]

        # 利用插值法获取所有cycle的容量
        cycle = np.arange(1,self.cycle_life+1)
        try:
            f = interpolate.interp1d(test_capacity_cycles,test_capacity,kind='cubic',fill_value='extrapolate')
        except:
            f = interpolate.interp1d(test_capacity_cycles,test_capacity,kind='linear',fill_value='extrapolate')
        degradation_trajectory = f(cycle)
        return degradation_trajectory #,(np.array(test_capacity_cycles),test_capacity)

    def get_capacity(self):
        '''
        获取电池的容量曲线
        :return:
        '''
        capacity = self.summary[0][0][1].reshape(-1)
        return capacity

    def get_value(self,cycle,variable):
        '''
        从cycle中提取出variable的数据
        :param cycle: int: 电池的循环次数
        :param variable: int or str: 变量的名称或者索引,可选self.variable_name中的变量
        :return:
        '''
        if isinstance(variable,str):
            variable = self.variable_name.index(variable)
        assert cycle <= self.data.shape[1]
        assert variable <= 7
        value = self.data[0][cycle-1][variable]
        if variable == 7:
            value = value[0]
        else:
            value = value.reshape(-1)
        return value

    # 如果需要重采样，则取消下面这行注释
    # @interpolate_resample(resample=False,num_points=128)
    def get_partial_value(self,cycle,variable,stage=1):
        '''
        从cycle中提取出variable的stage阶段的数据
        :param cycle: int: 电池的循环次数
        :param variable: int or str: 变量的名称或者索引,可选self.variable_name中的变量
        :param stage: int: 阶段的索引，可选[1,2,3,4], 分别是【充电，静置，放电，静置】; 对于Batch6模拟卫星的实验数据，一共有三个阶段[1,2,3]，【分别是充电，静置，放电】
        :return:
        '''
        value = self.get_value(cycle=cycle,variable=variable)
        relative_time_min = self.get_value(cycle=cycle,variable='relative_time_min')
        # 找到relative_time_min中等于0的index
        index = np.where(relative_time_min == 0)[0]
        index = np.insert(index,len(index),len(value))
        value = value[index[stage-1]:index[stage]]
        return value

    # 如果需要重采样，则取消下面这行注释
    # @interpolate_resample(resample=False,num_points=128)
    def get_CC_value(self, cycle, variable, voltage_range=None):
        '''
        获取第cycle个充电的周期的CC过程中的variable的值;如果指定了voltage_range,那么就是在voltage_range范围内的值
        :param cycle: int: 电池的循环次数
        :param variable: int or str: 变量的名称或者索引,可选self.variable_name中的变量
        :param voltage_range: list or None: 电压的范围,默认为None,表示整个CC过程的数据. 也可选其他范围，for example:[3.5,4.0]
        :return:
        '''
        value = self.get_partial_value(cycle=cycle, variable=variable, stage=1)
        voltage = self.get_partial_value(cycle=cycle, variable='voltage_V', stage=1)
        if voltage_range is None:
            index = np.where(voltage <= 4.199)[0]
        else:
            index = np.where((voltage >= voltage_range[0]) & (voltage <= voltage_range[1]))[0]
        value = value[index]
        return value

    # 如果需要重采样，则取消下面这行注释
    # @interpolate_resample(resample=False,num_points=128)
    def get_CV_value(self, cycle, variable, current_range=None):
        '''
        获取第cycle个充电的周期的CV过程中的variable的值;如果指定了current_range,那么就是在current_range范围内的值
        :param cycle: int: 电池的循环次数
        :param variable: int or str: 变量的名称或者索引,可选self.variable_name中的变量
        :param current_range: list or None: 电流的范围,默认为None,表示整个CV过程的数据. 其他可选：for example:[1.0,0.5]
        :return:
        '''
        value = self.get_partial_value(cycle=cycle, variable=variable, stage=1)
        current = self.get_partial_value(cycle=cycle, variable='current_A', stage=1)
        voltage = self.get_partial_value(cycle=cycle, variable='voltage_V', stage=1)
        if current_range is None:
            index = np.where(voltage >= 4.199)[0]
        else:
            index = np.where((current >= np.min(current_range)) & (current <= np.max(current_range)))[0]
        value = value[index]
        return value

    def get_one_cycle(self,cycle):
        '''
        获取某个cycle的所有通道数据
        :param cycle: int: 电池的循环次数
        :return:
        '''
        assert cycle <= self.data.shape[1]
        cycle_data = {}
        cycle_data['system_time'] = self.get_value(cycle=cycle,variable='system_time')
        cycle_data['relative_time_min'] = self.get_value(cycle=cycle,variable='relative_time_min')
        cycle_data['voltage_V'] = self.get_value(cycle=cycle,variable='voltage_V')
        cycle_data['current_A'] = self.get_value(cycle=cycle,variable='current_A')
        cycle_data['capacity_Ah'] = self.get_value(cycle=cycle,variable='capacity_Ah')
        cycle_data['power_Wh'] = self.get_value(cycle=cycle,variable='power_Wh')
        cycle_data['temperature_C'] = self.get_value(cycle=cycle,variable='temperature_C')
        cycle_data['description'] = self.get_value(cycle=cycle,variable='description')
        return cycle_data

    def get_IC_curve1(self,cycle,voltage_range=[3.6,4.19],step_len=0.01):
        '''
        计算增量容量曲线，公式为：dQ/dV,其中Q为容量，V为电压
        :param cycle: int: 电池的循环次数
        :param voltage_range: list: 电压的范围,默认为None,表示整个电池的电压范围
        :param step_len: float: 对容量数据进行等电压的间隔重采样，默认电压间隔为0.01V
        :return:
        '''
        Q = self.get_CC_value(cycle=cycle,variable='capacity_Ah',voltage_range=voltage_range)
        V = self.get_CC_value(cycle=cycle,variable='voltage_V',voltage_range=voltage_range)

        if len(Q) <= 2 or len(V) <= 2:
            return [],[]

        # 对Q进行等V间隔重采样
        f1 = interpolate.interp1d(V, Q, kind='linear')
        num_points = int((voltage_range[1] - voltage_range[0]) / step_len) + 1
        V_new = np.linspace(V[0], V[-1], num_points)
        Q_new = f1(V_new)

        dQ = np.diff(Q_new)
        dV = np.diff(V_new)
        IC = dQ/dV

        return IC,V_new[1:]

    def get_IC_curve2(self,cycle,voltage_range=[3.6,4.19],step_len=0.01):
        '''
        计算增量容量曲线，公式为：dQ/dV = I·dt/dV
        :param cycle: int: 电池的循环次数
        :param voltage_range: list: 电压的范围,默认为None,表示整个电池的电压范围
        :param step_len: float: 对电流和时间数据进行等电压的间隔重采样，默认电压间隔为0.01V
        :return:
        '''
        t = self.get_CC_value(cycle=cycle,variable='relative_time_min',voltage_range=voltage_range)
        V = self.get_CC_value(cycle=cycle,variable='voltage_V',voltage_range=voltage_range)
        I = self.get_CC_value(cycle=cycle,variable='current_A',voltage_range=voltage_range)

        # 对t和I进行等电压V间隔重采样
        num_points = int((voltage_range[1] - voltage_range[0]) / step_len) + 1
        f1 = interpolate.interp1d(V, t, kind='linear')
        V_new = np.linspace(V[0], V[-1], num_points)
        t_new = f1(V_new)
        f2 = interpolate.interp1d(V, I, kind='linear')
        I_new = f2(V_new)

        dt = np.diff(t_new)
        dV = np.diff(V_new)
        Idt = I_new[1:]*dt
        IC = Idt/dV
        return IC,V_new[1:]


if __name__ == '__main__':
    # 一个简单的例子
    battery_path = r'..\Batch-1\2C_battery-1.mat'
    battery = Battery(battery_path)
    IC1,V1 = battery.get_IC_curve1(cycle=10,voltage_range=[3.6,4.19])

    import matplotlib.pyplot as plt
    plt.plot(V1,IC1)
    plt.show()
