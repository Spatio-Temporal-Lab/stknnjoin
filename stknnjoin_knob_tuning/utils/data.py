import random
import datetime
import time
import warnings

import numpy as np
import pandas as pd
from numpy import histogram, histogram2d
from scipy import stats
from geopy.distance import geodesic
from pyproj import Geod
from shapely.geometry import Point, LineString, Polygon


class Data:
    def __init__(self, data_path):
        self.data_path = data_path

    # 获取数据长度
    def get_len(self):
        length = 0
        with open(self.data_path) as f:
            for i, _ in enumerate(f):
                length = i
            print(length)

        return length

    def iter_count(self):
        from itertools import (takewhile, repeat)
        buffer = 1024 * 1024
        with open(self.data_path) as f:
            buf_gen = takewhile(lambda x: x, (f.read(buffer) for _ in repeat(None)))
            return sum(buf.count('\n') for buf in buf_gen)

    # 统计数据特征信息(len, time_range, spatial_range)
    def data_info(self):
        df = pd.read_csv(self.data_path, header=None, sep=',', skiprows=lambda i: i > 0 and random.random() > 0.02)

        df['time'] = df[3].map(time_format)
        df['longitude'] = df[4].map(longitude_format)
        df['latitude'] = df[4].map(latitude_format)

        timerange = (df['time'].max() - df['time'].min()).days
        longitude_max, longitude_min = df['longitude'].max(), df['longitude'].min()
        longitude_range = longitude_max - longitude_min
        latitude_max, latitude_min = df['latitude'].max(), df['latitude'].min()
        latitude_range = latitude_max - latitude_min

        sp_dist = self.spatial_distribution(df['longitude'], df['latitude'],
                                            [[longitude_min, longitude_max], [latitude_min, latitude_max]], 5)
        tp_dist = self.temporal_distribution(df['time'])
        return len(df), timerange, longitude_range, latitude_range, sp_dist, tp_dist

    def data_info_line(self):
        def get_gps_list(line_string, x_list, y_list):
            pair_list = line_string[12:-1].split(', ')
            for pair in pair_list:
                longitude = pair.split(' ')[0]
                latitude = pair.split(' ')[1]
                x_list.append(longitude)
                y_list.append(latitude)
            return

        # LINESTRING (108.99675 34.25279, 108.99675 34.25322, 108.99672 34.25364)	2018-10-06 09:43:13	2018-10-06 09:43:22
        df = pd.read_csv(self.data_path, header=None, sep='	', skiprows=lambda i: i > 0 and random.random() > 0.02)

        df['time_s'] = df[1].map(time_format)
        df['time_e'] = df[2].map(time_format)

        longitude_list = []
        latitude_list = []
        df[0].apply(get_gps_list, x_list=longitude_list, y_list=latitude_list)

        timerange = (df['time'].max() - df['time'].min()).days
        longitude_max = np.max(longitude_list)
        longitude_min = np.min(longitude_list)
        latitude_max = np.max(latitude_list)
        latitude_min = np.min(latitude_list)

        longitude_range = longitude_max - longitude_min
        latitude_range = latitude_max - latitude_min

        sp_dist = self.spatial_distribution(longitude_list, latitude_list,
                                            [[longitude_min, longitude_max], [latitude_min, latitude_max]], 5)
        tp_dist = self.temporal_distribution(df['time'])
        return len(df), timerange, longitude_range, latitude_range, sp_dist, tp_dist, len(longitude_list) / len(df), 0

    def data_info_polygon(self):
        def get_gps_list(polygon_string, x_list, y_list):
            pair_list = polygon_string[10:-2].split(', ')
            for pair in pair_list:
                longitude = pair.split(' ')[0]
                latitude = pair.split(' ')[1]
                x_list.append(longitude)
                y_list.append(latitude)
            return

        # POLYGON ((108.94704 34.24012, 108.94694 34.24095, 108.94703 34.24072, 108.94704 34.24012))	2018-11-18 21:13:43	2018-11-18 21:15:46
        df = pd.read_csv(self.data_path, header=None, sep='	', skiprows=lambda i: i > 0 and random.random() > 0.02)

        df['time_s'] = df[1].map(time_format)
        df['time_e'] = df[2].map(time_format)

        longitude_list = []
        latitude_list = []
        df[0].apply(get_gps_list, x_list=longitude_list, y_list=latitude_list)

        time_range = (df['time'].max() - df['time'].min()).days
        longitude_max = np.max(longitude_list)
        longitude_min = np.min(longitude_list)
        latitude_max = np.max(latitude_list)
        latitude_min = np.min(latitude_list)

        longitude_range = longitude_max - longitude_min
        latitude_range = latitude_max - latitude_min

        sp_dist = self.spatial_distribution(longitude_list, latitude_list,
                                            [[longitude_min, longitude_max], [latitude_min, latitude_max]], 5)
        tp_dist = self.temporal_distribution(df['time'])
        return len(df), time_range, longitude_range, latitude_range, sp_dist, tp_dist, len(longitude_list) / len(df), 0

    # point数据转存csv
    def save_csv_by_line(self, out):
        items = []
        with open(self.data_path) as f:
            for i, line in enumerate(f):
                date, longitude, latitude = self.parse_point(line)
                items.append([date, longitude, latitude])
        df = pd.DataFrame(items)
        df.columns = ['date', 'longitude', 'latitude']
        df.to_csv(out, index=False)

    # 解析point数据
    def parse_point(self, line):
        items = line.split(',')
        time_str = items[3]
        # 2013-05-02
        # date = time.strptime(time_str.split(' ')[0], '%Y-%m-%d')
        date = datetime.datetime.strptime(time_str.split(' ')[0], '%Y-%m-%d').date()
        point = items[4]
        pair = point.split('(')[1].replace(')', '')
        longitude = float(pair.split(' ')[0])
        latitude = float(pair.split(' ')[1])

        return date, longitude, latitude

    # 解析line数据
    def parse_line(self, line):
        line = line.replace('LINESTRING (', '').replace(')', '')
        items = line.split('	')
        points = items[0]
        time_start = items[1]
        time_end = items[2]
        date_start = datetime.datetime.strptime(time_start.split(' ')[0], '%Y-%m-%d').date()
        date_end = datetime.datetime.strptime(time_end.split(' ')[0], '%Y-%m-%d').date()

        point_list = points.split(', ')
        longitude_list = []
        latitude_list = []
        for pair in point_list:
            lo = float(pair.split(' ')[0])
            la = float(pair.split(' ')[1])
            longitude_list.append(lo)
            latitude_list.append(la)

        return date_start, date_end, longitude_list, latitude_list

    # 解析polygon数据
    def parse_polygon(self, line):
        line = line.replace('POLYGON ((', '').replace('))', '')
        items = line.split('\t')
        points = items[0]
        time_start = items[1]
        time_end = items[2]
        date_start = datetime.datetime.strptime(time_start.split(' ')[0], '%Y-%m-%d').date()
        date_end = datetime.datetime.strptime(time_end.split(' ')[0], '%Y-%m-%d').date()

        point_list = points.split(', ')
        longitude_list = []
        latitude_list = []
        for pair in point_list:
            lo = float(pair.split(' ')[0])
            la = float(pair.split(' ')[1])
            longitude_list.append(lo)
            latitude_list.append(la)

        return date_start, date_end, longitude_list, latitude_list

    # 从总数据中随机获取部分数据
    def get_partial_data(self, nums=4000000, total=8000000, out='../resources/chengdu/point/result'):
        index_list = random.sample(range(0, total), nums)
        index_list.sort()

        file_result = open(out, 'a')
        with open(self.data_path) as f:
            i = 0
            # while i <= len(index_list):
            for j, line in enumerate(f):
                if j == index_list[i]:
                    file_result.write(f'{line}')
                    i += 1
                if i >= len(index_list):
                    break
        file_result.close()

    # 从总数据中随机获取两组数据（r和s）
    def get_partial_data_double(self,
                                nums_r=4000000,
                                nums_s=4000000,
                                total=631988343,
                                out_r='../resources/chengdu/point2/point_r',
                                out_s='../resources/chengdu/point2/point_s'):
        index_list = random.sample(range(0, total), nums_r + nums_s)
        index_list_r = random.sample(index_list, nums_r)
        index_list_s = list(set(index_list) - set(index_list_r))
        print(len(index_list))
        print(len(index_list_r))
        print(len(index_list_s))
        index_list_r.sort()
        index_list_s.sort()

        file_result_r = open(out_r, 'a')
        file_result_s = open(out_s, 'a')
        with open(self.data_path) as f:
            i = 0
            j = 0
            for k, line in enumerate(f):
                if i < len(index_list_r) and k == index_list_r[i]:
                    file_result_r.write(line)
                    i += 1
                if j < len(index_list_s) and k == index_list_s[j]:
                    file_result_s.write(line)
                    j += 1
                if (i >= len(index_list_r)) and (j >= len(index_list_s)):
                    break
        file_result_r.close()
        file_result_s.close()

    # 数据分布（时、空独立）
    def relative_freq(self, a, numbins=10):
        freq_result = stats.relfreq(a, numbins=numbins)
        return freq_result.frequency, freq_result.lowerlimit, freq_result.binsize

    # 数据空间分布
    def spatial_distribution(self, longitude_list, latitude_list, area, bins=5):
        sp_dist, _, _ = histogram2d(longitude_list, latitude_list, bins=bins, range=area)
        return sp_dist

    # 数据时间分布
    def temporal_distribution(self, date_list):
        tp_dist = pd.value_counts(date_list, sort=False)
        return tp_dist


# format fucntion
def time_format(x):
    return datetime.datetime.strptime(x[:10], '%Y-%m-%d').date()


def longitude_format(x):
    pair = x[7:-1]
    longitude = float(pair.split(' ')[0])
    return longitude


def latitude_format(x):
    pair = x[7:-1]
    latitude = float(pair.split(' ')[1])
    return latitude


# 从不同文件中提取数据(point)
def get_partial_data_from_multi_files(nums, m, out='../data/point/point'):
    file_total = [14437282, 13637679, 15365225, 14743749, 14386065, 14030360]
    for i in range(m):
        data = Data(f'../data/point/point_r_{i + 1}')
        data.get_partial_data(nums // m, file_total[i], f'{out}_r_{int(nums / 10000)}w_{m * 30}d')

        data2 = Data(f'../data/point/point_s_{i + 1}')
        data2.get_partial_data(nums // m, file_total[i], f'{out}_s_{int(nums / 10000)}w_{m * 30}d')


# 从不同文件中提取数据(double, line)
def get_partial_data_from_multi_files_line(nums_r, nums_s, t, out='../data/line'):
    file_total = [11550836, 34685823, 30560394, 32371352]
    file_name = ['line_1001_1015_seg_b', 'line_1015_1031_seg_b', 'line_1101_1105_seg_b', 'line_1115_1130_seg_b']
    for i in range(t):
        file = file_name[i]
        data = Data(f'../data/line/{file}')
        data.get_partial_data_double(nums_r // t, nums_s // t, file_total[i],
                                     f'{out}/line_r_{int(nums_r / 10000)}w_{t * 15}d',
                                     f'{out}/line_s_{int(nums_s / 10000)}w_{t * 15}d')


# 从不同文件中提取数据(polygon)
def get_partial_data_from_multi_files_polygon(nums_r, nums_s, t, out='../data/polygon'):
    # file_total = [803362, 2468042, 2468042, 2363305]
    file_total = 400000
    file_name_r = ['data1001_40w_r', 'data1015_40w_r', 'data1101_40w_r', 'data1115_40w_r']
    file_name_s = ['data1001_40w_s', 'data1015_40w_s', 'data1101_40w_s', 'data1115_40w_s']
    for i in range(t):
        file_r = file_name_r[i]
        file_s = file_name_s[i]
        data = Data(f'../data/polygon/{file_r}')
        data.get_partial_data(nums_r // t, file_total, f'{out}/polygon_r_{int(nums_r / 10000)}w_{t * 15}d')
        data2 = Data(f'../data/polygon/{file_s}')
        data2.get_partial_data(nums_s // t, file_total, f'{out}/polygon_s_{int(nums_s / 10000)}w_{t * 15}d')


# 加入时间分布
def get_feature_point(w, d):
    # 5w_15d
    file_r = f'../data/point/point_r_{w}_{d}'
    file_s = f'../data/point/point_s_{w}_{d}'
    data_r = Data(file_r)
    data_s = Data(file_s)
    nums_r, timerange_r, longitude_range_r, latitude_range_r, sp_dist_r, tp_dist_r = data_r.data_info()
    nums_s, timerange_s, longitude_range_s, latitude_range_s, sp_dist_s, tp_dist_s = data_s.data_info()

    record = pd.read_csv(f'../log/point/point_{w}_{w}_{d}/data_info', sep='\t')
    cols = get_cols('point')
    record.columns = cols
    for i, rec in enumerate(zip(tp_dist_r, tp_dist_s)):
        r = rec[0]
        s = rec[1]
        record.insert(29 + i, f'sp_r_{i}', r)
        record.insert(58 + 2 * i + 1, f'sp_s_{i}', s)

    record.to_csv(f'../log/point/point_{w}_{w}_{d}/record_3', sep='\t', header=False, index=False)


def get_feature_line(w, d):
    # 5w_15d
    file_r = f'../data/line/line_r_{w}_{d}'
    file_s = f'../data/line/line_s_{w}_{d}'
    data_r = Data(file_r)
    data_s = Data(file_s)
    nums_r, timerange_r, longitude_range_r, latitude_range_r, sp_dist_r, tp_dist_r, avg_point_r, distance_r = data_r.data_info_line()
    nums_s, timerange_s, longitude_range_s, latitude_range_s, sp_dist_s, tp_dist_s, avg_point_s, distance_s = data_s.data_info_line()

    record = pd.read_csv(f'../log/line/line_{w}_{w}_{d}/data_info', sep='\t')
    cols = get_cols('line')
    record.columns = cols
    for i, rec in enumerate(zip(tp_dist_r, tp_dist_s)):
        r = rec[0]
        s = rec[1]
        record.insert(29 + i, f'sp_r_{i}', r)
        record.insert(58 + 2 * i + 1, f'sp_s_{i}', s)

    record.to_csv(f'../log/line/line_{w}_{w}_{d}/record_2', sep='\t', header=False, index=False)


def get_feature_polygon(w, d):
    # 5w_15d
    file_r = f'../data/polygon/polygon_r_{w}_{d}'
    file_s = f'../data/polygon/polygon_s_{w}_{d}'
    data_r = Data(file_r)
    data_s = Data(file_s)
    nums_r, timerange_r, longitude_range_r, latitude_range_r, sp_dist_r, tp_dist_r, avg_point_r, area_r = data_r.data_info_polygon()
    nums_s, timerange_s, longitude_range_s, latitude_range_s, sp_dist_s, tp_dist_s, avg_point_s, area_s = data_s.data_info_polygon()

    record = pd.read_csv(f'../log/polygon/polygon_{w}_{w}_{d}/data_info', sep='\t')
    cols = get_cols('line')
    record.columns = cols
    for i, rec in enumerate(zip(tp_dist_r, tp_dist_s)):
        r = rec[0]
        s = rec[1]
        record.insert(29 + i, f'sp_r_{i}', r)
        record.insert(58 + 2 * i + 1, f'sp_s_{i}', s)

    record.to_csv(f'../log/polygon/polygon_{w}_{w}_{d}/record_2', sep='\t', header=False, index=False)


def get_feature_lp(w, d):
    # 5w_15d
    file_r = f'../data/line/line_r_{w}_{d}'
    file_s = f'../data/polygon/polygon_s_{w}_{d}'
    data_r = Data(file_r)
    data_s = Data(file_s)
    nums_r, timerange_r, longitude_range_r, latitude_range_r, sp_dist_r, tp_dist_r, avg_point_r, area_r = data_r.data_info_line()
    nums_s, timerange_s, longitude_range_s, latitude_range_s, sp_dist_s, tp_dist_s, avg_point_s, area_s = data_s.data_info_polygon()

    record = pd.read_csv(f'../log/lp/lp_{w}_{w}_{d}/data_info', sep='\t')
    cols = get_cols('polygon')
    record.columns = cols
    for i, rec in enumerate(zip(tp_dist_r, tp_dist_s)):
        r = rec[0]
        s = rec[1]
        record.insert(29 + i, f'sp_r_{i}', r)
        record.insert(58 + 2 * i + 1, f'sp_s_{i}', s)

    record.to_csv(f'../log/lp/lp_{w}_{w}_{d}/record_2', sep='\t', header=False, index=False)


def get_cols(data_type):
    if data_type == 'point':
        cols = ['nums_r', 'timerange_r', 'longitude_range_r', 'latitude_range_r']
        for i in range(25):
            cols.append(f'sp_dist_r_{i + 1}')
        for i in range(31):
            cols.append(f'tp_dist_r_{i + 1}')
        cols.extend(['nums_s', 'timerange_s', 'longitude_range_s', 'latitude_range_s'])
        for i in range(25):
            cols.append(f'sp_dist_s_{i + 1}')
        for i in range(31):
            cols.append(f'tp_dist_s_{i + 1}')

        cols.extend(['k', 'nodes', 'alpha', 'beta', 'binNum', 'execute_time'])

    elif data_type == 'line':
        cols = ['nums_r', 'timerange_r', 'longitude_range_r', 'latitude_range_r', 'avg_p_r', 'dist_r']
        for i in range(25):
            cols.append(f'sp_dist_r_{i + 1}')
        for i in range(31):
            cols.append(f'tp_dist_r_{i + 1}')
        cols.extend(['nums_s', 'timerange_s', 'longitude_range_s', 'latitude_range_s', 'avg_p_s'])
        for i in range(25):
            cols.append(f'sp_dist_s_{i + 1}')
        for i in range(31):
            cols.append(f'tp_dist_s_{i + 1}')
        cols.extend(['k', 'nodes', 'alpha', 'beta', 'binNum', 'execute_time'])

    elif data_type == 'polygon':
        cols = ['nums_r', 'timerange_r', 'longitude_range_r', 'latitude_range_r', 'avg_p_r']
        for i in range(25):
            cols.append(f'sp_dist_r_{i + 1}')
        for i in range(31):
            cols.append(f'tp_dist_r_{i + 1}')
        cols.extend(['nums_s', 'timerange_s', 'longitude_range_s', 'latitude_range_s', 'avg_p_s'])
        for i in range(25):
            cols.append(f'sp_dist_s_{i + 1}')
        for i in range(31):
            cols.append(f'tp_dist_s_{i + 1}')
        cols.extend(['k', 'nodes', 'alpha', 'beta', 'binNum', 'execute_time'])

    elif data_type == 'lp':
        cols = ['nums_r', 'timerange_r', 'longitude_range_r', 'latitude_range_r', 'avg_p_r']
        for i in range(25):
            cols.append(f'sp_dist_r_{i + 1}')
        for i in range(32):
            cols.append(f'tp_dist_r_{i + 1}')
        cols.extend(['nums_s', 'timerange_s', 'longitude_range_s', 'latitude_range_s', 'avg_p_s'])
        for i in range(25):
            cols.append(f'sp_dist_s_{i + 1}')
        for i in range(32):
            cols.append(f'tp_dist_s_{i + 1}')
        cols.extend(['k', 'nodes', 'alpha', 'beta', 'binNum', 'execute_time'])
    return cols


def get_all_feature(data_type):
    # point
    if data_type == 'point':
        for i in range(5, 35, 5):
            get_feature_point(f'{i}w', '30d')
    if data_type == 'line':
        for i in range(5, 35, 5):
            get_feature_line(f'{i}w', '30d')
    if data_type == 'polygon':
        for i in range(5, 35, 5):
            get_feature_polygon(f'{i}w', '30d')
    if data_type == 'lp':
        for i in range(5, 35, 5):
            get_feature_lp(f'{i}w', '30d')


def change_date():
    df = pd.read_csv('../log/line/line_20w_20w_60d/record', sep='\t')
    df.columns = get_cols('line')
    df['timerange_r'] = 31
    df['timerange_s'] = 31

    file_r = f'../data/line/line_r_20w_30d'
    file_s = f'../data/line/line_s_20w_30d'
    data_r = Data(file_r)
    data_s = Data(file_s)
    nums_r, timerange_r, longitude_range_r, latitude_range_r, sp_dist_r, tp_dist_r, avg_point_r, distance_r = data_r.data_info_line()
    nums_s, timerange_s, longitude_range_s, latitude_range_s, sp_dist_s, tp_dist_s, avg_point_s, distance_s = data_s.data_info_line()
    for i, rec in enumerate(zip(tp_dist_r, tp_dist_s)):
        r = rec[0]
        s = rec[1]
        df.insert(29 + i, f'sp_r_{i}', r)
        df.insert(58 + 2 * i + 1, f'sp_s_{i}', s)

    df.to_csv(f'../log/line/line_20w_20w_60d/record_2', sep='\t', header=False, index=False)


def change_date2(file):
    df = pd.read_csv(f'../log/point/{file}/record_a', sep='\t')
    df.columns = get_cols('point')
    df['timerange_r'] = 30
    df['timerange_s'] = 30

    file_r = f'../data/point/point_r_5w_30d'
    file_s = f'../data/point/point_s_5w_30d'
    data_r = Data(file_r)
    data_s = Data(file_s)
    nums_r, timerange_r, longitude_range_r, latitude_range_r, sp_dist_r, tp_dist_r = data_r.data_info()
    nums_s, timerange_s, longitude_range_s, latitude_range_s, sp_dist_s, tp_dist_s = data_s.data_info()
    for i, rec in enumerate(zip(tp_dist_r, tp_dist_s)):
        r = rec[0]
        s = rec[1]
        df.insert(29 + i, f'sp_r_{i}', r)
        df.insert(58 + 2 * i + 1, f'sp_s_{i}', s)

    df.to_csv(f'../log/point/{file}/record_a2', sep='\t', header=False, index=False)



if __name__ == '__main__':
    # data = Data('../resources/chengdu/point_s')
    # data.get_partial_data(419544 * 2, 41954474, out='../resources/chengdu/point/point_s_80w')

    # data = Data('../resources/chengdu/point/point_r_40w')
    # nums, timerange, longitude_range, latitude_range = data.data_info()
    # print(nums, timerange, longitude_range, latitude_range)
    # data.save_csv_by_line('../resources/chengdu/point/point_r_40w.csv')
    # random.seed(2022)
    # data = Data('../data/polygon/polygon_1115')

    # data.get_len()
    # data.get_partial_data_double()
    # data_r.get_partial_data(50000 // 3, 13637679, '../data/point/point_r_5w_90d')
    # get_partial_data_from_multi_files_line(150000, 150000, 2)
    # get_partial_data_from_multi_files(100000, 3)
    # data_s = Data('../data/point/point_s_3')
    # data_s.get_partial_data(50000 // 3, 13637679, '../data/point/point_s_5w_90d')
    # data = Data('../resources/chengdu/point2/point_r')
    # data.save_csv_by_line('../resources/chengdu/point2/point_r.csv')

    # point
    # get_partial_data_from_multi_files(300000, 1)

    # polygon
    # for i in range(1, 5):
    # get_partial_data_from_multi_files_polygon(300000, 300000, 2)

    # line 
    # for i in range(1, 5):
    # get_partial_data_from_multi_files_line(200000, 200000, 2)

    # get_all_feature('point')
    change_date2('point_5w_5w_30d')
