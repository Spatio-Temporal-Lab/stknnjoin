import joblib
import pandas as pd
import numpy as np
import os
import argparse
import pickle
import time

import skopt
from matplotlib import pyplot as plt

from Executor import Executor
from datetime import datetime

from predict import Model, Data_model
from utils.RuntimeContext import RuntimeContext
from utils.data import Data, get_cols
from utils.logger import TrainingLogger
from skopt import gp_minimize, forest_minimize
from skopt.plots import plot_convergence, plot_evaluations, plot_objective

from utils.plots import plot_rf

# 获取参数
args = RuntimeContext()
# 参数初始化设置
args.knobs_setting = {
    'alpha': 100,
    'beta': 20,
    'binNum': 200
}


# args.data_type = args.output_dir.split('/')[1]

def st_knn_join(x):
    executor = Executor(args=args)

    # 为参数负值
    args.knobs_setting['alpha'] = x[0]
    args.knobs_setting['beta'] = x[1]
    args.knobs_setting['binNum'] = x[2]

    time_sum = 0
    logger.print(
        f'alpha: {args.knobs_setting["alpha"]}, beta: {args.knobs_setting["beta"]}, binNum: {args.knobs_setting["binNum"]}')

    # 可以执行n次，取平均结果
    cnt = 3
    for i in range(cnt):
        file_save_path = f'{args.save_dir}/{args.knobs_setting["alpha"]}_{args.knobs_setting["beta"]}_{args.knobs_setting["binNum"]}/{i}'
        if os.system(f'hdfs dfs -test -e {file_save_path}/part-00000') != 0:
            executor.execute(alpha=args.knobs_setting["alpha"], beta=args.knobs_setting["beta"],
                             binNum=args.knobs_setting["binNum"], save_path=file_save_path)
        time = executor.parse_result_hdfs(file_save_path)
        logger.print(f'time: {time}')
        time_sum += time
    logger.print(f'avg_time: {time_sum / cnt}\n')
    if args.data_type == 'point':
        record(time_sum / cnt)
    elif args.data_type == 'line':
        record_line(time_sum / cnt)
    elif args.data_type == 'polygon':
        record_polygon(time_sum / cnt)
    elif args.data_type == 'lp':
        record_lp(time_sum / cnt)
    return time_sum / cnt


# online模式，使用预测模型
def st_knn_join_online(x):
    executor = Executor(args=args)
    st_knn_join(x)

    # 为参数负值
    if args.knob == 'all':
        args.knobs_setting['alpha'] = x[0]
        args.knobs_setting['beta'] = x[1]
        args.knobs_setting['binNum'] = x[2]
    else:
        args.knobs_setting[args.knob] = x[0]

    logger.print('======predict======')
    logger.print(f'alpha: {args.knobs_setting["alpha"]}, beta: {args.knobs_setting["beta"]}, '
                 f'binNum: {args.knobs_setting["binNum"]}')

    features = args.features.copy()
    for knob in ['alpha', 'beta', 'binNum']:
        # items.append(args.knobs_setting[knob])
        features.loc[0, knob] = args.knobs_setting[knob]
    int_cols = ['nums_r', 'nums_s', 'alpha', 'beta', 'binNum']
    features[int_cols] = features[int_cols].apply(pd.to_numeric, downcast='integer', errors='ignore')
    features[features.columns.difference(int_cols)] = features[features.columns.difference(int_cols)].apply(
        pd.to_numeric, downcast='float', errors='ignore')
    result = args.model.predict(features)
    logger.print(f'pred_time: {result[0]}')
    return result[0]


def search_online():
    """
    online贝叶斯搜索，加入预测模块
    :return:
    [alpha, beta, binNum]
    """
    # 记录初始化时间
    init_time = time.time()
    # log文件夹下record的数据训练模型
    if args.data_type == 'point':
        data_model = Data_model(data_path='data_model/point_w_tp', args=args)
    elif args.data_type == 'line':
        data_model = Data_model(data_path='data_model/line_w_tp', args=args)
    elif args.data_type == 'polygon':
        data_model = Data_model(data_path='data_model/polygon_w_tp', args=args)
    elif args.data_type == 'lp':
        data_model = Data_model(data_path='data_model/lp_w_tp', args=args)
    model = Model()
    model.train(data_model.data_train)
    pred = model.predict(data_model.data_test.drop(['execute_time'], axis='columns'))
    model.evaluate(pred, data_model.data_test['execute_time'])
    args.model = model

    data_r = Data(args.data_file_r_local)
    data_s = Data(args.data_file_s_local)
    if args.data_type == 'point':
        nums_r, timerange_r, longitude_range_r, latitude_range_r, sp_dist_r, tp_dist_r = data_r.data_info()
        nums_s, timerange_s, longitude_range_s, latitude_range_s, sp_dist_s, tp_dist_s = data_s.data_info()
        items = [nums_r, timerange_r, longitude_range_r, latitude_range_r]
        for dist in sp_dist_r:
            for x in dist:
                items.append(x)
        items.extend(tp_dist_r[:31])
        items.extend([nums_s, timerange_s, longitude_range_s, latitude_range_s])
        for dist in sp_dist_s:
            for x in dist:
                items.append(x)
        items.extend(tp_dist_s[:31])
    elif args.data_type in ['line', 'polygon', 'lp']:
        nums_r, timerange_r, longitude_range_r, latitude_range_r, sp_dist_r, tp_dist_r, avg_p_r = data_r.data_info_line()
        nums_s, timerange_s, longitude_range_s, latitude_range_s, sp_dist_s, tp_dist_s, avg_p_s = data_s.data_info_line()
        items = [nums_r, timerange_r, longitude_range_r, latitude_range_r, avg_p_r]
        for dist in sp_dist_r:
            for x in dist:
                items.append(x)
        items.extend(tp_dist_r)
        items.extend([nums_s, timerange_s, longitude_range_s, latitude_range_s, avg_p_s])
        for dist in sp_dist_s:
            for x in dist:
                items.append(x)
        items.extend(tp_dist_s)

    k = args.k
    items.append(k)
    items.extend([0, 0, 0])
    features = pd.DataFrame(columns=get_cols(args.data_type)[:-1])

    features.loc[0] = items
    features.drop(['k', 'timerange_s'], axis='columns', inplace=True)
    args.features = features
    # 记录开始搜索时间
    start_time = time.time()
    args.logger.print(f'init_time: {(start_time - init_time)}\n')
    res = gp(st_knn_join_online, 100, 60)
    # 记录搜索结束时间
    end_time = time.time()
    args.logger.print(f'search_time: {(end_time - start_time)}\n')

    return res


def search_outline():
    """
    outline贝叶斯搜索，传统搜索模式，记录搜索过程用于训练预测模型
    :return:
    [alpha, beta, binNum]
    """
    start_time = datetime.now()
    res = gp(st_knn_join, 10, 5)
    end_time = datetime.now()
    args.logger.print(f'search_time: {(end_time - start_time).seconds}')
    return res


def draw(res):
    noise_level = 0.1

    for n_iter in range(5):
        # Plot true function.
        plt.subplot(5, 2, 2 * n_iter + 1)

        if n_iter == 0:
            show_legend = True
        else:
            show_legend = False

        ax = plot_rf(res, n_calls=n_iter,
                     objective=st_knn_join,
                     noise_level=noise_level,
                     show_legend=show_legend, show_title=False,
                     show_next_point=False, show_acq_func=False)
        ax.set_ylabel("")
        ax.set_xlabel("")

        # Plot EI(x)
        plt.subplot(5, 2, 2 * n_iter + 2)
        ax = plot_rf(res, n_calls=n_iter,
                     show_legend=show_legend, show_title=False,
                     show_mu=False, show_acq_func=True,
                     show_observations=False,
                     show_next_point=True)
        ax.set_ylabel("")
        ax.set_xlabel("")

    plt.savefig(f'{args.save_path}/function.png')
    plt.show()


# gp回归
def gp(function=st_knn_join, n_calls=25, n_random_starts=15):
    # 设置三个参数范围
    bound = [(5, 1000), (1, 100), (1, 1000)]
    # initial (optional)
    # x0, y0 = get_x0_y0('log/point/point_3m/record')

    gp_start = datetime.now()
    res = gp_minimize(function,  # the function to minimize
                      bound,  # the bounds on each dimension of x
                      acq_func="EI",  # the acquisition function
                      n_calls=n_calls,  # the number of evaluations of f
                      n_initial_points=n_random_starts,  # the number of random initialization points
                      # noise=0.1 ** 2,  # the noise level (optional)
                      # random_state=2023,    # the random seed
                      # x0=x0,
                      # y0=y0,
                      n_jobs=-1)
    gp_end = datetime.now()
    args.logger.print(f'gp_time: {(gp_end - gp_start).seconds}\n')
    # 保存
    with open(f'{args.output_dir}/{args.knob}.pkl', 'wb') as f:
        pickle.dump(res, f)
    fig = plot_convergence(res).get_figure()
    fig.savefig(f'{args.output_dir}/output.png')
    fig2 = plot_evaluations(res)
    plt.savefig(f'{args.output_dir}/output2.png')
    fig3 = plot_objective(res)
    plt.savefig(f'{args.output_dir}/output3.png')

    return res


# 记录每一次执行的数据特征、参数以及对应执行时间，存储在历史仓库中(point)
def record(execute_time, target_file=f'{args.output_dir}/record'):
    data_r = Data(args.data_file_r_local)
    data_s = Data(args.data_file_s_local)
    nums_r, timerange_r, longitude_range_r, latitude_range_r, sp_dist_r, tp_dist_r = data_r.data_info()
    nums_s, timerange_s, longitude_range_s, latitude_range_s, sp_dist_s, tp_dist_s = data_s.data_info()
    k = args.k
    nodes = args.nodes
    with open(target_file, 'a') as f:
        # 写入特征
        f.write(f'{nums_r}\t{timerange_r}\t{longitude_range_r}\t{latitude_range_r}\t')
        for sp_d_r in sp_dist_r:
            for x in sp_d_r:
                f.write(f'{int(x)}\t')
        for tp_d_r in tp_dist_r[:31]:
            f.write(f'{int(tp_d_r)}\t')
        f.write(f'{nums_s}\t{timerange_s}\t{longitude_range_s}\t{latitude_range_s}\t')
        for sp_d_s in sp_dist_s:
            for y in sp_d_s:
                f.write(f'{int(y)}\t')
        for tp_d_s in tp_dist_s[:31]:
            f.write(f'{int(tp_d_s)}\t')
        f.write(f'{k}\t{nodes}\t')

        for knob in ['alpha', 'beta', 'binNum']:
            f.write(f'{args.knobs_setting[knob]}\t')
        f.write(str(execute_time))
        f.write('\n')


# 记录每一次执行的数据特征、参数以及对应执行时间，存储在历史仓库中(line)
def record_line(execute_time, target_file=f'{args.output_dir}/record'):
    data_r = Data(args.data_file_r_local)
    data_s = Data(args.data_file_s_local)
    nums_r, timerange_r, longitude_range_r, latitude_range_r, sp_dist_r, tp_dist_r, avg_point_r = data_r.data_info_line()
    nums_s, timerange_s, longitude_range_s, latitude_range_s, sp_dist_s, tp_dist_s, avg_point_s = data_s.data_info_line()
    k = args.k
    nodes = args.nodes
    with open(f'{target_file}', 'a') as f:
        # 写入特征
        f.write(f'{nums_r}\t{timerange_r}\t{longitude_range_r}\t{latitude_range_r}\t{avg_point_r}\t')
        for sp_d_r in sp_dist_r:
            for x in sp_d_r:
                f.write(f'{int(x)}\t')
        for tp_r in tp_dist_r:
            f.write(f'{int(tp_r)}\t')
        f.write(f'{nums_s}\t{timerange_s}\t{longitude_range_s}\t{latitude_range_s}\t{avg_point_s}\t')
        for sp_d_s in sp_dist_s:
            for y in sp_d_s:
                f.write(f'{int(y)}\t')
        for tp_s in tp_dist_s:
            f.write(f'{int(tp_s)}\t')
        f.write(f'{k}\t{nodes}\t')

        for knob in ['alpha', 'beta', 'binNum']:
            f.write(f'{args.knobs_setting[knob]}\t')
        f.write(str(execute_time))
        f.write('\n')


# 记录每一次执行的数据特征、参数以及对应执行时间，存储在历史仓库中(polygon)
def record_polygon(execute_time, target_file=f'{args.output_dir}/record'):
    data_r = Data(args.data_file_r_local)
    data_s = Data(args.data_file_s_local)
    nums_r, timerange_r, longitude_range_r, latitude_range_r, sp_dist_r, tp_dist_r, avg_point_r = data_r.data_info_polygon()
    nums_s, timerange_s, longitude_range_s, latitude_range_s, sp_dist_s, tp_dist_s, avg_point_s = data_s.data_info_polygon()
    k = args.k
    nodes = args.nodes
    with open(target_file, 'a') as f:
        # 写入特征
        f.write(f'{nums_r}\t{timerange_r}\t{longitude_range_r}\t{latitude_range_r}\t{avg_point_r}\t')
        for sp_d_r in sp_dist_r:
            for x in sp_d_r:
                f.write(f'{int(x)}\t')
        for tp_r in tp_dist_r:
            f.write(f'{int(tp_r)}\t')
        f.write(f'{nums_s}\t{timerange_s}\t{longitude_range_s}\t{latitude_range_s}\t{avg_point_s}\t')
        for sp_d_s in sp_dist_s:
            for y in sp_d_s:
                f.write(f'{int(y)}\t')
        for tp_s in tp_dist_s:
            f.write(f'{int(tp_s)}\t')
        f.write(f'{k}\t{nodes}')

        for knob in ['alpha', 'beta', 'binNum']:
            f.write(f'{args.knobs_setting[knob]}\t')
        f.write(str(execute_time))
        f.write('\n')


# 记录每一次执行的数据特征、参数以及对应执行时间，存储在历史仓库中(lp)
def record_lp(execute_time, target_file=f'{args.output_dir}/record'):
    data_r = Data(args.data_file_r_local)
    data_s = Data(args.data_file_s_local)
    nums_r, timerange_r, longitude_range_r, latitude_range_r, sp_dist_r, tp_dist_r, avg_point_r = data_r.data_info_line()
    nums_s, timerange_s, longitude_range_s, latitude_range_s, sp_dist_s, tp_dist_s, avg_point_s = data_s.data_info_polygon()
    k = args.k
    nodes = args.nodes
    with open(target_file, 'a') as f:
        # 写入特征
        f.write(f'{nums_r}\t{timerange_r}\t{longitude_range_r}\t{latitude_range_r}\t{avg_point_r}\t')
        for sp_d_r in sp_dist_r:
            for x in sp_d_r:
                f.write(f'{int(x)}\t')
        for st_r in tp_dist_r:
            f.write(f'{int(st_r)}\t')
        f.write(f'{nums_s}\t{timerange_s}\t{longitude_range_s}\t{latitude_range_s}\t{avg_point_s}\t')
        for sp_d_s in sp_dist_s:
            for y in sp_d_s:
                f.write(f'{int(y)}\t')
        for st_s in tp_dist_s:
            f.write(f'{int(st_s)}\t')
        f.write(f'{k}\t')

        for knob in ['alpha', 'beta', 'binNum']:
            f.write(f'{args.knobs_setting[knob]}\t')
        f.write(str(execute_time))
        f.write('\n')


# 记录数据特征及选择最优参数结果
def record_final_result(res, real_time):
    if args.data_type == 'point':
        for i, knob in enumerate(['alpha', 'beta', 'binNum']):
            args.logger.print(f'{knob}: {res.x[i]}\t')
        # args.logger.print(str(real_time))
        record(real_time, 'data_model/point_w_tp')
    elif args.data_type == 'line':
        for i, knob in enumerate(['alpha', 'beta', 'binNum']):
            args.logger.print(f'{knob}: {res.x[i]}\t')
        # args.logger.print(str(real_time))
        record_line(real_time, 'data_model/line_w_tp')
    elif args.data_type == 'polygon':
        for i, knob in enumerate(['alpha', 'beta', 'binNum']):
            args.logger.print(f'{knob}: {res.x[i]}\t')
        # args.logger.print(str(real_time))
        record_polygon(real_time, 'data_model/polygon_w_tp')
    elif args.data_type == 'lp':
        for i, knob in enumerate(['alpha', 'beta', 'binNum']):
            args.logger.print(f'{knob}: {res.x[i]}\t')
        # args.logger.print(str(real_time))
        record_lp(real_time, 'data_model/lp_w_tp')


# 验证实际执行时间
def get_real_time(knob_list):
    st_knn_join(knob_list)


def get_x0_y0(fp):
    df = pd.read_csv(fp, header=None)
    x0 = df.loc[:, 119:121].values.tolist()  # alpha, beta, binNum
    y0 = df.loc[:, 122].values.tolist()  # execute time
    return x0, y0


# 执行bayes优化
if __name__ == '__main__':
    log_path = f'{args.output_dir}'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    logger = TrainingLogger(f'{log_path}/log')
    args.logger = logger
    args.save_path = log_path

    r = search_outline()
    # r = search_online()

    for knob in r.x:
        args.logger.print(f'{knob}\t')
    args.logger.print(str(r.fun))
    real_time = st_knn_join(r.x)
    args.logger.print(f'real_time: {real_time}')
    # res = gp()
    # draw(res)
    record_final_result(r, real_time)
