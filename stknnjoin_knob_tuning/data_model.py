import pandas as pd
from sklearn.model_selection import train_test_split

from utils.get_columns import get_cols


class Data_Model:
    def __init__(self, data_path, data_type='point'):
        self.data_path = data_path
        self.data_type = data_type

        # 将数据保存为DataFrame
        df = pd.read_csv(self.data_path, sep='\t', header=None)
        self.data = df
        self.set_header()

        # self.feature_selection()
        self.data_type_to_numeric()

        self.x = self.data.drop(columns='execute_time')
        self.y = self.data.loc[:, 'execute_time']

        # 划分训练训练集测试集
        # self.data_train, self.data_test = train_test_split(df, test_size=0.2)

    # 设置dataframe的header
    def set_header(self):
        self.data.columns = get_cols(self.data_type)

    # 特征选择
    def feature_selection(self):
        data_type = self.data_type
        if data_type == 'point':
            unused_cols = ['k', 'timerange_s']
        elif data_type == 'line':
            unused_cols = ['k', 'timerange_s', 'dist_r', 'dist_s']
        elif data_type == 'polygon':
            unused_cols = ['k', 'timerange_s', 'area_r', 'area_s']
        elif data_type == 'lp':
            unused_cols = ['k', 'timerange_s', 'dist_r', 'area_s']

        self.data.drop(unused_cols, axis='columns', inplace=True)

    # 设置数据类型
    def data_type_to_numeric(self):
        float_cols = ['longitude_range_r', 'latitude_range_r', 'longitude_range_s', 'latitude_range_s', 'execute_time']
        self.data[float_cols] = self.data[float_cols].apply(pd.to_numeric, downcast='float', errors='ignore')
        int_cols = self.data.columns.difference(float_cols)
        self.data[int_cols] = self.data[int_cols].apply(pd.to_numeric, downcast='integer', errors='ignore')

    def data_info(self):
        # print(self.data.dtypes)

        # pd.to_numeric(data_cp, downcast='float')
        print(self.data.describe())
        print(self.data.columns)
