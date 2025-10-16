'''
Author: Fujin Wang
Date: 2025/10
Github: https://github.com/wang-fujin
Description: 

Description:
该代码用于读取CALCE公开的电池数据集，方便后续预处理和分析。
数据集链接：https://calce.umd.edu/battery-data#CS2

包括以下这些电池：
CS33, CS34, CS35, CS36, CS37, CS38,
CX33, CX34, CX35, CX36, CX37, CX38
'''


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import openpyxl





class Loader:
    '''
    读取一个文件夹中的所有xlsx文件，并将它们合并成一个dataframe
    '''
    def __init__(self, folder='../../CALCE数据/CS2/type2/CS2_35'):
        self.folder = folder
        self.files = self._sort_file_by_time()
        self.df = self.merge_xlsx_file()
        print('--'*100)
        print('df.shape:', self.df.shape)
        print('columns:', self.df.columns)

    def _sort_file_by_time(self):
        '''
        每个CS2_xx文件夹中有多个文件，获取文件夹名字，并按时间排序
        :return:
        '''
        files = os.listdir(self.folder)
        files = sorted(files, key=lambda x: os.path.getmtime(os.path.join(self.folder, x)))

        print('该电池数据集一共包含{}个文件'.format(len(files)))
        return files

    def _get_sheet_names(self, path):
        '''
        获取xlsx文件中的sheet名称
        :param path: xlsx文件路径
        :return: list: sheet名称
        '''
        channel = 'Channel_1'  # xlsx文件中有Info、Channel_、Statistics_ 等sheet，只需要读取Channel_的sheet
        wb = openpyxl.load_workbook(path, read_only=True, keep_links=False)
        sheet_names = wb.sheetnames
        wb.close()

        needed_sheet_names = []
        for i in range(len(sheet_names)):
            this_sheet = wb[sheet_names[i]]
            if channel in this_sheet.title:
                print('--第' + str(i + 1) + '个sheet Name: ' + this_sheet.title)
                needed_sheet_names.append(this_sheet.title)
        return needed_sheet_names

    def read_one_xlsx(self, path):
        '''
        读取一个xlsx文件，返回一个dataframe
        :param path: xlsx文件路径
        :return:
        '''
        xlsx_path = os.path.join(self.folder, path)
        sheet_name = self._get_sheet_names(xlsx_path)
        if len(sheet_name) == 1:
            data = pd.read_excel(xlsx_path, sheet_name=sheet_name[0], usecols=[2, 3, 4, 5, 6, 7, 8, 9])
            print(f'--包含一张表：{sheet_name[0]}')
        elif len(sheet_name) == 2:
            df1 = pd.read_excel(xlsx_path, sheet_name=sheet_name[0], usecols=[2, 3, 4, 5, 6, 7, 8, 9])
            df2 = pd.read_excel(xlsx_path, sheet_name=sheet_name[1], usecols=[2, 3, 4, 5, 6, 7, 8, 9])
            data = pd.concat([df1, df2])
            data.index = range(data.shape[0])
            print(f'--包含两张张表：{sheet_name[0]}')
            print(f'    -------- {sheet_name[1]}')
        else:
            print('--None')
            return None
        return data

    def merge_xlsx_file(self):
        '''
        把所有xlsx合并成为一个dataframe
        :return:
        '''
        All_dfs = []
        for file in self.files:
            print(f'正在读取文件：{file}')
            df_i = self.read_one_xlsx(file)
            All_dfs.append(df_i)
        df = pd.concat(All_dfs, ignore_index=True)
        df['Date_Time'] = pd.to_datetime(df['Date_Time'])
        df.sort_values(by='Date_Time', inplace=True)
        return df

class Battery:
    def __init__(self,path):
        parent_dir = os.path.dirname(path)
        battery_name = os.path.basename(path)
        # 判断是否存在hdf5文件,存在则直接读取；如果不存在，则读取Excel文件处理并保存为hdf5文件
        hdf5_path = os.path.join(#parent_dir + '/' +
                                    battery_name + '.hdf5')
        if os.path.exists(hdf5_path):
            self.df = pd.read_hdf(hdf5_path, key=battery_name)
        else:
            loader = Loader(path)
            self.df = loader.df

            self.df.to_hdf(hdf5_path, key=battery_name, mode='w')

        self.variable_name = ['Date_Time', 'Step_Time(s)', 'Step_Index', 'Cycle_Index', 'Current(A)',
       'Voltage(V)', 'Charge_Capacity(Ah)', 'Discharge_Capacity(Ah)']
        self.preprocess_df()
        self.cycle_index = np.unique(self.df['Cycle_Index'].values)
        self.cycle_life = len(self.cycle_index)

        print('-'*40,f' Battery {battery_name} ','-'*40)
        print('电池寿命：',self.cycle_life)
        print('data shape:',self.df.shape)
        print('变量名:',self.variable_name)
        print('-'*100)

    def preprocess_df(self):
        '''
        对数据进行预处理，重新计算Cycle_Index
        :return:
        '''
        # 重新计算Cycle_Index, 对原有的列进行累加得到信的Cycle_Index
        diff = self.df['Cycle_Index'].diff()
        reset_points = diff < 0
        new_index = self.df['Cycle_Index'].copy()

        # 初始化
        cumulative_offset = 0
        prev_index = 0

        for i in np.where(reset_points)[0]:
            # 当前段最大值（上一段的最大值）
            segment_max = self.df['Cycle_Index'][prev_index:i].max()
            cumulative_offset = segment_max
            new_index[i:] += cumulative_offset

            # 更新上一段起点
            prev_index = i

        self.df['Cycle_Index'] = new_index

    def _check(self,cycle=None,variable=None):
        '''
        检查输入的cycle和variable是否合法
        :param cycle: int: 循环次数
        :param variable: str: 变量名
        :return: bool: 是否合法
        '''
        if cycle is not None:
            if cycle not in self.df['Cycle_Index'].values:
                raise ValueError('cycle should be in [{},{}]'.format(int(self.df['Cycle_Index'].min()),int(self.df['Cycle_Index'].max())))
        if variable is not None:
            if variable not in self.variable_name:
                raise ValueError('variable should be in {}'.format(self.variable_name))
        return True

    def get_cycle(self,cycle):
        '''
        获取第cycle次循环的数据
        :param cycle: int: 循环次数
        :return: DataFrame
        '''
        self._check(cycle=cycle)
        cycle_df = self.df[self.df['Cycle_Index']==cycle]
        return cycle_df

    def get_value(self,cycle,variable):
        '''
        获取第cycle次循环的某个变量的值
        :param cycle: 循环次数
        :param variable: 变量名
        :return:
        '''
        self._check(cycle=cycle,variable=variable)
        cycle_df = self.get_cycle(cycle)
        return cycle_df[variable].values

    def get_charge_data(self,cycle):
        '''
        获取第cycle次循环的充电数据
        :param cycle:
        :return:
        '''
        self._check(cycle=cycle)
        cycle_df = self.get_cycle(cycle)
        # 充电数据的电流为正值
        charge_data = cycle_df[cycle_df['Current(A)']>0.01]
        return charge_data

    def get_discharge_data(self,cycle):
        '''
        获取第cycle次循环的放电数据
        :param cycle:
        :return:
        '''
        self._check(cycle=cycle)
        cycle_df = self.get_cycle(cycle)
        # 放电数据的电流为负值
        discharge_data = cycle_df[cycle_df['Current(A)']<-0.01]
        return discharge_data

    def get_CC_charge(self,cycle, voltage_range=None):
        '''
        获取第cycle次循环的恒流充电数据，可以指定电压范围
        :param cycle: 循环次数
        :param voltage_range: tuple或list或None: 电压范围，默认为None表示整个CC阶段; 其他形如[3.8,4.0]
        :return:
        '''
        self._check(cycle=cycle)
        charge_data = self.get_charge_data(cycle)
        # 恒流充电阶段，电压小于4.2V
        if voltage_range is None:
            CC_charge_data = charge_data[charge_data['Voltage(V)'] < 4.199]
        else:
            index = np.where((charge_data['Voltage(V)'] > voltage_range[0]) &
                              (charge_data['Voltage(V)'] < voltage_range[1]))[0]
            CC_charge_data = charge_data.iloc[index]
        return CC_charge_data

    def get_CV_charge(self,cycle, current_range=None):
        '''
        获取第cycle次循环的恒压充电数据，可以指定电流范围
        :param cycle: 循环次数
        :param current_range: tuple或list或None: 电流范围，默认为None表示整个CV阶段; 其他形如[1.0,0.1]
        :return:
        '''
        self._check(cycle=cycle)
        charge_data = self.get_charge_data(cycle)
        CV_charge_data = charge_data[charge_data['Voltage(V)'] > 4.199]
        if current_range is None:
            return CV_charge_data
        else:
            index = np.where((CV_charge_data['Current(A)'] > min(current_range)) &
                              (CV_charge_data['Current(A)'] < max(current_range)))[0]
            CV_charge_data = CV_charge_data.iloc[index]
            return CV_charge_data



def plot_one_cycle(cycle_df):
    '''
    画一个cycle的曲线
    :param cycle_df:
    :return:
    '''
    fig,ax = plt.subplots(2,4,figsize=(12,6))
    for i in range(8):
        row = i // 4
        col = i % 4
        colname  = cycle_df.columns[i]
        ax[row,col].scatter(cycle_df.index,cycle_df[colname])
        ax[row,col].set_title(colname)
    plt.tight_layout()
    plt.show()




if __name__ == '__main__':
    folder = '../../CALCE数据/CS2/type2/CS2_35'
    battery = Battery(folder)

    cycle_df = battery.get_discharge_data(5)
    plot_one_cycle(cycle_df)

