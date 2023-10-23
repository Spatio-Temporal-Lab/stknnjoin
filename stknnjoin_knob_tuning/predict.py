import pandas as pd
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error, explained_variance_score, mean_absolute_error, mean_absolute_percentage_error, make_scorer
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import lightgbm
import xgboost
import numpy as np



class Model:
    def __init__(self, args=None):
        # self.model = DecisionTreeRegressor()
        # self.model = LinearRegression()
        # self.model = KNeighborsRegressor()
        # self.model = SVR()
        # self.model = Lasso()
        # self.model = Ridge()
        # self.model = MLPRegressor()
        # self.model = RandomForestRegressor()
        # self.model = AdaBoostRegressor()
        # self.model = GradientBoostingRegressor()
        # self.model = BaggingRegressor()
        # self.model = lightgbm.LGBMRegressor()
        self.model = xgboost.XGBRegressor()
        # self.model = xgboost.XGBRFRegressor()
        self.args = args

    def train(self, data_train):
        x_train = data_train.drop(['execute_time'], axis='columns')
        y_train = data_train['execute_time']
        self.model.fit(x_train, y_train)

    def predict(self, x):
        pred = self.model.predict(x)

        return pred

    # 验证
    # todo：10-fold交叉
    def evaluate(self, pred, y_true):
        """
        r2_score：判定系数，其含义是也是解释回归模型的方差得分，其值取值范围是[0,1]，越接近于1说明自变量越能解释因
        变量的方差变化，值越小则说明效果越差。
        mean_squared_error：均方差（Mean squared error，MSE），该指标计算的是拟合数据和原始数据对应样本点的误差的
        平方和的均值，其值越小说明拟合效果越好。
        mean_absolute_error：平均绝对误差（Mean Absolute Error，MAE），用于评估预测结果和真实数据集的接近程度的程度
        ，其其值越小说明拟合效果越好。
        explained_variance_score：解释回归模型的方差得分，其值取值范围是[0,1]，越接近于1说明自变量越能解释因变量
        的方差变化，值越小则说明效果越差。
        """
        r2 = r2_score(y_true, pred)
        mse = mean_squared_error(y_true, pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, pred)
        mape = mean_absolute_percentage_error(y_true, pred)
        evs = explained_variance_score(y_true, pred)
        
        # self.args.logger.print(f'r2: {r2}')
        # self.args.logger.print(f'mse: {mse}')
        # self.args.logger.print(f'mae: {mae}')
        print(r2, mse, rmse, mae, mape, evs)
        return r2, mse, mae, evs

    def cross_val(self, x, y):
        cv = ShuffleSplit(n_splits=5, test_size=0.2)
        mse = cross_val_score(self.model, x, y, scoring=make_scorer(mean_squared_error), cv=cv)
        rmse = np.sqrt(mse)
        mae = cross_val_score(self.model, x, y, scoring=make_scorer(mean_absolute_error), cv=cv)
        mape = cross_val_score(self.model, x, y, scoring=make_scorer(mean_absolute_percentage_error), cv=cv)
        print('mse: ', mse, np.mean(mse))
        print('rmse: ', rmse, np.mean(rmse))
        print('mae: ', mae, np.mean(mae))
        print('mape: ', mape, np.mean(mape))
        return mse, rmse, mae, mape

class Data_model:
    def __init__(self, data_path='data_model/record_point', args=None):
        self.data_path = data_path
        self.args = args

        # 将数据保存为DataFrame
        df = pd.read_csv(self.data_path, sep='\t', header=None)
        self.data = df
        self.set_header()

        self.feature_selection()
        self.data_type_to_numeric()

        self.x = self.data.drop(columns='execute_time')
        self.y = self.data.loc[:, 'execute_time']

        # 划分训练训练集测试集
        self.data_train, self.data_test = train_test_split(df, test_size=0.2)

    # 添加表头(弃，读入DataFrame时再设置表头)
    def add_header(self):
        with open(self.data_path, 'r+') as f:
            content = f.read()
            f.seek(0, 0)
            f.write(f'nums_r\ttimerange_r\tlongitude_range_r\tlowerlimit_lo_r\tbinsize_lo_r\t')
            for i in range(10):
                f.write(f'fr_lo_r_{i}\t')
            f.write(f'latitude_range_r\tlowerlimit_la_r\tbinsize_la_r\t')
            for i in range(10):
                f.write(f'fr_la_r_{i}\t')

            f.write(f'nums_s\ttimerange_s\tlongitude_range_s\tlowerlimit_lo_s\tbinsize_lo_s\t')
            for i in range(10):
                f.write(f'fr_lo_s_{i}\t')
            f.write(f'latitude_range_s\tlowerlimit_la_s\tbinsize_la_s\t')
            for i in range(10):
                f.write(f'fr_la_s_{i}\t')

            f.write(f'k\t')

            for knob in ['alpha', 'beta', 'binNum']:
                f.write(f'{knob}\t')
            f.write('execute_time\n')
            f.write(content)

    # 设置dataframe的header
    def set_header(self):
        self.data.columns = get_cols(self.args.data_type)

    # 特征选择
    def feature_selection(self):
        # unused_cols = ['k', 'timerange_s']
        # unused_cols = ['k', 'timerange_s', 'dist_r', 'dist_s']
        unused_cols = ['k', 'timerange_s', 'area_r', 'area_s']
        # unused_cols = ['k', 'timerange_s', 'dist_r', 'area_s']
        self.data.drop(unused_cols, axis='columns', inplace=True)

    # 设置数据类型
    def data_type_to_numeric(self):
        float_cols = ['longitude_range_r', 'latitude_range_r', 'longitude_range_s', 'latitude_range_s', 'execute_time']
        self.data[float_cols] = self.data[float_cols].apply(pd.to_numeric, downcast='float', errors='ignore')
        self.data[self.data.columns.difference(float_cols)] = self.data[self.data.columns.difference(float_cols)].apply(pd.to_numeric, downcast='integer', errors='ignore')

    def data_info(self):
        # print(self.data.dtypes)

        # pd.to_numeric(data_cp, downcast='float')
        print(self.data.describe())
        print(self.data.columns)


def get_cols(data_type):
    print(data_type)
    if data_type == 'point':
        cols = ['nums_r', 'timerange_r', 'longitude_range_r', 'latitude_range_r']
        for i in range(25):
            cols.append(f'sp_dist_r_{i + 1}')
        for i in range(31):
            cols.append(f'tp_dist_r_{i+1}')
        cols.extend(['nums_s', 'timerange_s', 'longitude_range_s', 'latitude_range_s'])
        for i in range(25):
            cols.append(f'sp_dist_s_{i + 1}')
        for i in range(31):
            cols.append(f'tp_dist_s_{i + 1}')

        cols.extend(['k', 'alpha', 'beta', 'binNum', 'execute_time'])
    
    elif data_type == 'line':
        cols = ['nums_r', 'timerange_r', 'longitude_range_r', 'latitude_range_r', 'avg_p_r', 'dist_r']
        for i in range(25):
            cols.append(f'sp_dist_r_{i + 1}')
        for i in range(32):
            cols.append(f'tp_dist_r_{i+1}')
        cols.extend(['nums_s', 'timerange_s', 'longitude_range_s', 'latitude_range_s', 'avg_p_s', 'dist_s'])
        for i in range(25):
            cols.append(f'sp_dist_s_{i + 1}')
        for i in range(32):
            cols.append(f'tp_dist_s_{i+1}')
        cols.extend(['k', 'alpha', 'beta', 'binNum', 'execute_time'])
    
    elif data_type == 'polygon':
        cols = ['nums_r', 'timerange_r', 'longitude_range_r', 'latitude_range_r', 'avg_p_r', 'area_r']
        for i in range(25):
            cols.append(f'sp_dist_r_{i + 1}')
        for i in range(32):
            cols.append(f'tp_dist_r_{i+1}')
        cols.extend(['nums_s', 'timerange_s', 'longitude_range_s', 'latitude_range_s', 'avg_p_s', 'area_s'])
        for i in range(25):
            cols.append(f'sp_dist_s_{i + 1}')
        for i in range(32):
            cols.append(f'tp_dist_s_{i+1}')
        cols.extend(['k', 'alpha', 'beta', 'binNum', 'execute_time'])
    elif data_type == 'lp':
        cols = ['nums_r', 'timerange_r', 'longitude_range_r', 'latitude_range_r', 'avg_p_r', 'dist_r']
        for i in range(25):
            cols.append(f'sp_dist_r_{i + 1}')
        for i in range(32):
            cols.append(f'tp_dist_r_{i+1}')
        cols.extend(['nums_s', 'timerange_s', 'longitude_range_s', 'latitude_range_s', 'avg_p_s', 'area_s'])
        for i in range(25):
            cols.append(f'sp_dist_s_{i + 1}')
        for i in range(32):
            cols.append(f'tp_dist_s_{i+1}')
        cols.extend(['k', 'alpha', 'beta', 'binNum', 'execute_time'])

    print('len_cols', len(cols))
    return cols


if __name__ == '__main__':
    # POINT
    class RC(object):
        def __init__(self):
            self.data_type = 'point'

    args = RC()
    data = Data_model('data_model/point_w_tp', args)
    # data.data_info()
    predict_model = Model()
    predict_model.cross_val(data.x, data.y)
    # predict_model.train(data.data_train)
    # pred = predict_model.predict(data.data_test.drop(['execute_time'], axis='columns'))
    # predict_model.evaluate(pred, data.data_test['execute_time'])
