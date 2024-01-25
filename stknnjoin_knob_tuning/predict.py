import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, explained_variance_score, mean_absolute_error, \
    mean_absolute_percentage_error, make_scorer
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit
import xgboost
import numpy as np

from utils.data import get_cols


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

        return r2, mse, mae, evs

    def cross_val(self, x, y):
        cv = ShuffleSplit(n_splits=5, test_size=0.2)
        mse = cross_val_score(self.model, x, y, scoring=make_scorer(mean_squared_error), cv=cv)
        rmse = np.sqrt(mse)
        mae = cross_val_score(self.model, x, y, scoring=make_scorer(mean_absolute_error), cv=cv)
        mape = cross_val_score(self.model, x, y, scoring=make_scorer(mean_absolute_percentage_error), cv=cv)
        return mse, rmse, mae, mape


class Data_model:
    def __init__(self, data_path='data_model/record_point', args=None):
        self.data_path = data_path
        self.args = args

        # 将数据保存为DataFrame
        df = pd.read_csv(self.data_path, sep='\t', header=get_cols(args.data_type))
        self.data = df

        self.drop_features()
        self.data_type_to_numeric()

        self.x = self.data.drop(columns='execute_time')
        self.y = self.data.loc[:, 'execute_time']

        # 划分训练训练集测试集
        self.data_train, self.data_test = train_test_split(df, test_size=0.2)

    def drop_features(self):
        unused_cols = ['k', 'timerange_s']
        self.data.drop(unused_cols, axis='columns', inplace=True)

    # 设置数据类型
    def data_type_to_numeric(self):
        float_cols = ['longitude_range_r', 'latitude_range_r', 'longitude_range_s', 'latitude_range_s', 'execute_time']
        self.data[float_cols] = self.data[float_cols].apply(pd.to_numeric, downcast='float', errors='ignore')
        self.data[self.data.columns.difference(float_cols)] = self.data[self.data.columns.difference(float_cols)].apply(
            pd.to_numeric, downcast='integer', errors='ignore')
