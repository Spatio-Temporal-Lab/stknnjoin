import lightgbm
import numpy as np
import xgboost
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, \
    explained_variance_score, make_scorer
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from data_model import Data_Model


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
        self.args = args

    def train(self, data_train):
        x_train = data_train.drop(['execute_time'], axis='columns')
        y_train = data_train['execute_time']
        self.model.fit(x_train, y_train)

    def predict(self, x):
        pred = self.model.predict(x)

        return pred

    # 评估
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


if __name__ == '__main__':
    # global_args._init()
    # global_args.set_value('data_type', 'point')
    data = Data_Model('data_model/point_w_tp')

    predict_model = Model()
    predict_model.cross_val(data.x, data.y)
