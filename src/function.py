import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns

#导入数据并检查字段是否齐全
def import_and_check_data(path, num_fields):
    """
    导入数据并检查字段是否齐全
    :param path: 数据路径
    :param num_fields: 数据字段数量
    :return data: DataFrame
    """
    data = pd.read_csv(path, low_memory=False)
    print("Data Shape:", data.shape)
    print("Data Columns:", data.columns)
    
    if len(data.columns) == num_fields:
        print("数据字段齐全")
    else:
        print("数据字段缺失")
    
    return data

#定义一个查看车辆属性的函数
def extract_vehicle_attributes(data: pd.DataFrame, attributes: list):
    """
    查看车辆属性
    :param data: DataFrame
    :param attributes: 车辆属性列表
    :return results: 车辆属性字典
    """
    
    results = {}
    for attribute in attributes:
        if attribute in data.columns:
            unique_values = data[attribute].unique()
            if len(unique_values) > 1:
                print(f"Attribute {attribute} is not unique.")
            elif pd.isnull(unique_values[0]):
                print(f"Attribute {attribute} is missing.")
            else:
                results[attribute] = unique_values[0]
        else:
            print(f"Attribute {attribute} does not exist in the data.")
    return results

#定义一个批量删除列的函数
def drop_columns(data: pd.DataFrame, columns_to_drop: list):
    data = data.drop(columns=columns_to_drop, errors='ignore')
    print("已删除列名为 {} 的列".format(columns_to_drop))
    return data

# 定义一个删除内部数据值完全相同的列的函数
def remove_duplicate_columns(data):
    """
    删除内部数据值完全相同的列
    :param data: DataFrame
    :return cleaned_data, result_df
    """

    same_col = []
    same_col_dict = []

    for col in data.columns:
        if len(data[col].unique()) == 1:
            same_col.append(col)
            same_col_dict.append(data[col].unique())
        else:
            pass
    
    cleaned_data = data.drop(same_col, axis=1)

    result_df = pd.DataFrame({'same_col': same_col, 'same_col_dict': same_col_dict})
    return cleaned_data, result_df

#定义一个删除重复数据的函数
def drop_duplicated(data):
    print('删除重复数据前：',data.shape)
    print('删除重复的行数为：',data.duplicated().sum())
    data.drop_duplicates(inplace=True)
    print('删除重复数据后：',data.shape)
    return data

#编写一个预处理时间的函数
def time_preprocessing(df,time_col='yr_modahrmn',format='%Y-%m-%d %H:%M:%S'):
    '''
    df:需要处理的数据集
    time_col:时间列的列名
    format:时间格式
    return:处理好的数据集
    '''
    df['time'] = pd.to_datetime(df[time_col],format = format)
    #删除原来的时间列
    df.drop(time_col,axis=1,inplace=True)
    #将时间列设置为索引
    df.set_index('time',inplace=True)
    #按照时间排序
    df.sort_index(inplace=True)
    #重置索引，使索引变为列
    df.reset_index(inplace=True)
    #时间跨度
    Timedelta = df['time'].iloc[-1] - df['time'].iloc[0]
    print('数据的总时间跨度为：',Timedelta)
    df['time_diff'] = df['time'].diff()
    #打印最小采样间隔
    print('最小采样间隔为：',df['time_diff'].min())
    #打印平均采样间隔
    print('平均采样间隔为：',df['time_diff'].mean())
    #打印最大采样间隔
    print('最大采样间隔为：',df['time_diff'].max())
    #删除time_diff列
    df.drop('time_diff',axis=1,inplace=True)
    #将时间列设置为索引
    df.set_index('time',inplace=True)
    return df

#编写一个分析特征的函数
def analyze_column(data, column_name,lower_bound = None,upper_bound = None):
    """
    分析特征
    :param data: DataFrame
    :param column_name: 特征名称
    :param upper_bound: 特征的上界
    :param lower_bound: 特征的下界
    :param ratio: 特征放缩比例
    :return:
    """
    # 查看列的数据类型
    print(f"数据类型为 {data[column_name].dtype}")
    

    # 如果数据类型为数字，查看列的统计信息
    if data[column_name].dtype in ['int64', 'float64', 'float32', 'int32']:
        print('-' * 50)
        print(data[column_name].describe())
        print('-' * 50)
        # 画出列的直方图
        data[column_name].hist(bins=50)
        plt.xlabel(column_name)
        plt.ylabel('Frequency')
        plt.title(f'Distribution of {column_name}')
        plt.show()
        print('-' * 50)
        # 画出列随时间的变化图
        plt.figure(figsize=(40,5))
        plt.plot(data.index, data[column_name])
        plt.xlabel('Time')
        plt.ylabel(column_name)
        plt.title(f'{column_name} over Time')
        plt.show()

    # 分析缺失值占比
    missing_number = data[column_name].isnull().sum()
    missing_ratio = missing_number / len(data) * 100

    #打印分割线
    print('-' * 50)
    print(f"'{column_name}' 有 {missing_number} 个缺失值")
    print(f"'{column_name}' 的缺失值占比为 {missing_ratio:.2f}%")

    # 查看列是否在有效值范围
    if upper_bound is not None and lower_bound is not None:
            #打印分割线
        print('-' * 50)
        print(f"有效值范围为 [{lower_bound}, {upper_bound}]")
        invalid_values = data[(data[column_name] < lower_bound) | (data[column_name] > upper_bound)]
        print(f"'{column_name}' 共有 {len(invalid_values)} 个范围外值")
    
    # 是否需要放缩
    # if ratio is not None:
    #     print('-' * 50)
    #     print(f"需要放缩，放缩比例为 {ratio}")
    #     data[column_name] = data[column_name].astype('float64')
    #     data[column_name] = data[column_name] * ratio
    #     print(f"列放缩完毕，数据类型为 {data[column_name].dtype}")

#档位数据处理
def parse_gear(gear):
    bit5 = (gear & 0b00100000) >> 5
    bit4 = (gear & 0b00010000) >> 4
    bits3210 = gear & 0b00001111

    gear_dict = {
        0b0000: 0,
        0b0001: 1,
        0b0010: 2,
        0b0011: 3,
        0b0100: 4,
        0b0101: 5,
        0b0110: 6,
        0b1101: 7,
        0b1110: 8,
        0b1111: 9,
    }
    gear_value = gear_dict.get(bits3210, np.nan)  # 如果为未知挡位，输出null
    return bit5, bit4, gear_value
    
import pandas as pd

#检查故障码
def decode_alarm(value):
    # 如果值为0，直接返回空列表
    if value == 0:
        return []

    # 将数值转换为二进制，并去除前面的 '0b'
    binary_representation = bin(value)[2:].zfill(32)  # 使用zfill确保长度为32位

    alarm_definitions = [
        "温度差异报警",
        "电池高温报警",
        "车载储能装置类型过压报警",
        "车载储能装置类型欠压报警",
        "SOC低报警",
        "单体电池过压报警",
        "单体电池欠压报警",
        "SOC过高报警",
        "SOC跳变报警",
        "可充电储能系统不匹配报警",
        "电池单体一致性差报警",
        "绝缘报警",
        "DC-DC温度报警",
        "制动系统报警",
        "DC-DC状态报警",
        "驱动电机控制器温度报警",
        "高压互锁状态报警",
        "驱动电机温度报警",
        "车载储能装置类型过充"
    ]

    active_alarms = []

    for idx, alarm in enumerate(alarm_definitions):
        # 从左侧开始检查，因为我们的列表是从低位到高位定义的
        if binary_representation[-(idx+1)] == '1':
            active_alarms.append(alarm)

    return active_alarms

#划分充电片段
def identify_charging_segments(data):
    """
    识别充电片段
    :param data: 包含充电状态的数据
    :return: 充电片段的开始时间和结束时间
    """
    segments = []
    start_time = None
    charging = False
    prev_time = None

    for current_time, row in data.iterrows():
        # Check for the start of charging
        if row['charging_status'] in [1, 2, 4] and not charging:
            start_time = current_time
            charging = True
        # Check for the end of charging or if there's a long time gap between records
        elif charging and (row['charging_status'] in [3] or (prev_time is not None and (current_time - prev_time).seconds > 1800)):
            segments.append((start_time, prev_time))
            charging = False

        prev_time = current_time

    if charging:
        segments.append((start_time, data.index[-1]))

    # Merge segments that are less than 5 minutes apart
    merged_segments = []
    for i, (start, end) in enumerate(segments):
        if i == 0:
            merged_segments.append((start, end))
        elif (start - merged_segments[-1][1]).seconds <= 300:
            merged_segments[-1] = (merged_segments[-1][0], end)
        else:
            merged_segments.append((start, end))

    return merged_segments

# def merge_segments(segments, merge_threshold=pd.Timedelta(minutes=5)):
#     """
#     合并充电片段
#     :param segments: 充电片段的开始时间和结束时间
#     :param merge_threshold: 合并阈值
#     :return: 合并后的充电片段
#     """
#     if not segments:
#         return []

#     merged_segments = [segments[0]]

#     for start, end in segments[1:]:
#         last_segment_end = merged_segments[-1][1]

#         if start - last_segment_end < merge_threshold:
#             merged_segments[-1] = (merged_segments[-1][0], end)
#         else:
#             merged_segments.append((start, end))

#     return merged_segments

def identify_driving_segments(data):
    """
    识别驾驶片段
    :param data: 包含车辆状态的数据
    :return: 驾驶片段的开始时间和结束时间
    """
    # Identify where the speed changes from or to zero
    data['speed_change'] = ((data['speed'] == 0) & (data['speed'].shift() != 0)) | ((data['speed'] != 0) & (data['speed'].shift() == 0))
    
    # Extract segments
    segments = []
    start_time = None
    for time, row in data.iterrows():
        if row['speed_change'] and row['speed'] != 0:
            start_time = time
        elif row['speed_change'] and row['speed'] == 0 and start_time is not None:
            end_time = time
            segments.append((start_time, end_time))
            start_time = None
    
    # Merge segments that are less than 10 minutes apart
    merged_segments = []
    prev_end = None
    for start, end in segments:
        if prev_end is None:
            merged_start = start
        elif (start - prev_end) <= timedelta(minutes=10):
            pass  # Continue with the segment
        else:
            merged_segments.append((merged_start, prev_end))
            merged_start = start
        prev_end = end
    if prev_end is not None:
        merged_segments.append((merged_start, prev_end))
    
        # Drop the 'speed_change' column
    data.drop(columns='speed_change', inplace=True)
    
    return merged_segments

def calculate_segment_details(data, segments):
    """
    计算片段的详细信息
    :param data: 包含充电状态和SOC的数据
    :param segments: 充电片段的开始时间和结束时间
    :return: 包含片段详细信息的DataFrame
    """
    segment_details = []

    for start_time, end_time in segments:
        segment_data = data[start_time:end_time]
        
        # Calculate the segment duration (in minutes)
        duration = (end_time - start_time).seconds / 60
        
        # Calculate the SOC change
        soc_change = segment_data['standard_soc'].iloc[-1] - segment_data['standard_soc'].iloc[0]
        
        segment_details.append({
            'start_time': start_time,
            'end_time': end_time,
            'duration': duration,
            'soc_change': soc_change
        })

    return pd.DataFrame(segment_details)

def process_string(data_string):
    """
    处理字符串
    :param data_string: 字符串
    :return: 整数列表
    """
    # 检查是否存在数字: 前缀，并去除
    if ":" in data_string:
        data_string = data_string.split(":")[1]
    
    data_list = data_string.split("_")  # 分割字符串
    # 如果字符串元素为空，替换为None，否则转换为整数
    data_list = [int(value) if value else None for value in data_list]  
    return data_list

def expand_columns(data):
    """
    展开列
    :param data: 包含列表的DataFrame
    :return: 展开后的DataFrame
    """
    # 定义函数，用于从列表中提取值
    def extract_values(lst, index):
        return lst[index] if index < len(lst) else None

    # 对cell_volt_list列进行处理
    volt_cols = len(max(data['cell_volt_list'], key=len))  # 获取最长的列表的长度
    volt_data = {}
    for i in range(volt_cols):
        col_name = f'cell_volt_{i}'
        volt_data[col_name] = data['cell_volt_list'].apply(extract_values, index=i)

    # 对cell_temp_list列进行处理
    temp_cols = len(max(data['cell_temp_list'], key=len))  # 获取最长的列表的长度
    temp_data = {}
    for i in range(temp_cols):
        col_name = f'cell_temp_{i}'
        temp_data[col_name] = data['cell_temp_list'].apply(extract_values, index=i)

    # 使用pd.concat合并所有新列
    expanded_data = pd.concat([data, pd.DataFrame(volt_data), pd.DataFrame(temp_data)], axis=1)
    
    return expanded_data

def plot_selected_columns(data, test_start_time, test_end_time, columns_to_plot):
    """
    绘制选定的列
    :param data: 包含所有列的DataFrame
    :param test_start_time: 测试开始时间
    :param test_end_time: 测试结束时间
    :param columns_to_plot: 要绘制的列
    """
    num_columns = len(columns_to_plot)
    fig, axs = plt.subplots(num_columns, 1, figsize=(20, 3*num_columns))
    
    for i, column in enumerate(columns_to_plot):
        axs[i].plot(data.loc[test_start_time:test_end_time][column])
        axs[i].set_ylabel(column)
    
    axs[-1].set_xlabel("Time")
    
    plt.tight_layout()
    plt.show()
