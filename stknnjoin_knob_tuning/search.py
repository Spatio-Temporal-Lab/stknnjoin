import os
import pickle
import time
from datetime import datetime

import pandas as pd
from matplotlib import pyplot as plt
from skopt import gp_minimize
from skopt.plots import plot_convergence, plot_evaluations, plot_objective

from data_model import Data_Model
from model import Model
from record import record
from stknn_executor import Executor
from utils.data import Data
from utils.get_columns import get_cols, get_features


def st_knn_join(x):
    executor = Executor(args_glb)

    # 为参数负值
    if args_glb.knob == 'all':
        args_glb.knobs_setting['alpha'] = x[0]
        args_glb.knobs_setting['beta'] = x[1]
        args_glb.knobs_setting['binNum'] = x[2]
    else:
        args_glb.knobs_setting[args_glb.knob] = x[0]

    time_sum = 0
    logger = args_glb.logger
    logger.print(
        f'alpha: {args_glb.knobs_setting["alpha"]}, beta: {args_glb.knobs_setting["beta"]}, binNum: {args_glb.knobs_setting["binNum"]}')
    # todo knobs设置

    # 可以执行n次，取平均结果
    cnt = 1
    for i in range(cnt):
        # file_save_path = f'./results_chengdu/line/{alpha}_{beta}_{binNum}/{i}'
        file_save_path = f'{args_glb.save_dir}/{args_glb.knobs_setting["alpha"]}_{args_glb.knobs_setting["beta"]}_{args_glb.knobs_setting["binNum"]}/{i}'
        # if not os.path.exists(file_save_path):
        if os.system(f'hdfs dfs -test -e {file_save_path}/part-00000') != 0:
            executor.execute(alpha=args_glb.knobs_setting["alpha"], beta=args_glb.knobs_setting["beta"],
                             binNum=args_glb.knobs_setting["binNum"], save_path=file_save_path)
        time = executor.parse_result(file_save_path)
        # time = executor.parse_result_hdfs(file_save_path)
        logger.print(f'time: {time}')
        time_sum += time
    logger.print(f'avg_time: {time_sum / cnt}\n')
    record(time_sum / cnt, args_glb)
    return time_sum / cnt

# online模式，使用预测模型
def st_knn_join_online(x):
    executor = Executor(args=args_glb)
    st_knn_join(x)

    # 为参数负值
    if args_glb.knob == 'all':
        args_glb.knobs_setting['alpha'] = x[0]
        args_glb.knobs_setting['beta'] = x[1]
        args_glb.knobs_setting['binNum'] = x[2]
    else:
        args_glb.knobs_setting[args_glb.knob] = x[0]

    logger = args_glb.logger
    logger.print('======predict======')
    logger.print(f'alpha: {args_glb.knobs_setting["alpha"]}, beta: {args_glb.knobs_setting["beta"]}, '
                 f'binNum: {args_glb.knobs_setting["binNum"]}')

    features = args_glb.features.copy()
    for knob in ['alpha', 'beta', 'binNum']:
        # items.append(args.knobs_setting[knob])
        features.loc[0, knob] = args_glb.knobs_setting[knob]
    int_cols = ['nums_r', 'nums_s', 'alpha', 'beta', 'binNum']
    features[int_cols] = features[int_cols].apply(pd.to_numeric, downcast='integer', errors='ignore')
    features[features.columns.difference(int_cols)] = features[features.columns.difference(int_cols)].apply(
        pd.to_numeric, downcast='float', errors='ignore')
    result = args_glb.model.predict(features)
    logger.print(f'pred_time: {result[0]}')
    return result[0]


def search_outline(args):
    """
    离线贝叶斯搜索，实际执行每一组参数。并记录搜索过程用于训练预测模型
    :return:
    [alpha, beta, binNum]
    """
    global args_glb
    args_glb = args
    start_time = datetime.now()
    res = gp(st_knn_join, 100, 60)
    end_time = datetime.now()

    logger = args.logger
    logger.print(f'search_time: {(end_time - start_time).seconds}')
    return res


def search_online(args):
    """
    在线贝叶斯搜索，加入预测模块，使用预测器输出值评估参数组效果
    :return:
    [alpha, beta, binNum]
    """
    global args_glb
    args_glb = args
    data_type = args_glb.data_type
    # 记录初始化时间
    init_time = time.time()
    # log文件夹下record的数据训练模型
    if data_type == 'point':
        data_model = Data_Model(data_path='data_model/point_w_tp')
    elif data_type == 'line':
        data_model = Data_Model(data_path='data_model/line_w_tp')
    elif data_type == 'polygon':
        data_model = Data_Model(data_path='data_model/polygon_w_tp')
    elif data_type == 'lp':
        data_model = Data_Model(data_path='data_model/lp_w_tp')
    model = Model()
    model.train(data_model.data)
    args_glb.model = model

    data_r = Data(args_glb.data_file_r_local)
    data_s = Data(args_glb.data_file_s_local)
    items = get_features(data_type)

    features = pd.DataFrame(columns=get_cols(data_type)[:-1])
    # print(len(features.columns))
    features.loc[0] = items
    unused_cols = ['k', 'timerange_s']
    # unused_cols = ['k', 'timerange_s', 'dist_r', 'dist_s']
    # unused_cols = ['k', 'timerange_s', 'area_r', 'area_s']
    # unused_cols = ['k', 'timerange_s', 'dist_r', 'area_s']
    features.drop(unused_cols, axis='columns', inplace=True)
    args_glb.features = features
    # 记录开始搜索时间
    start_time = time.time()
    args_glb.logger.print(f'init_time: {(start_time - init_time)}\n')
    res = gp(st_knn_join_online, 100, 60)
    # 记录搜索结束时间
    end_time = time.time()
    args_glb.logger.print(f'search_time: {(end_time - start_time)}\n')

    # 实际执行
    real_time = st_knn_join(res.x)
    args_glb.logger.print(f'real_time: {real_time}')

    return res

# gp回归
def gp(function=st_knn_join, n_calls=25, n_random_starts=15):
    logger = args_glb.logger
    # 设置三个参数范围
    if args_glb.knob == 'alpha':
        bound = [(50, 1000)]
    elif args_glb.knob == 'beta':
        bound = [(5, 100)]
    elif args_glb.knob == 'binNum':
        bound = [(10, 1000)]
    elif args_glb.knob == 'all':
        # 对alpha,beta,binNum同时建模
        bound = [(50, 1000), (5, 100), (10, 1000)]
    else:
        bound = []
        logger.print(f"knob err: {args_glb.knob}")
        return
    gp_start = datetime.now()
    res = gp_minimize(function,  # the function to minimize
                      bound,  # the bounds on each dimension of x
                      acq_func="EI",  # the acquisition function
                      n_calls=n_calls,  # the number of evaluations of f
                      n_initial_points=n_random_starts,  # the number of random initialization points
                      # noise=0.1 ** 2,  # the noise level (optional)
                      # random_state=2023,    # the random seed
                      x0=[100, 20, 200],
                      n_jobs=-1)
    gp_end = datetime.now()
    args_glb.logger.print(f'gp_time: {(gp_end - gp_start).seconds}\n')
    # 保存
    # skopt.dump(res, f'{args.save_path}/{args.knob}.pkl')

    with open(f'{args_glb.output_dir}/{args_glb.knob}.pkl', 'wb') as f:
        pickle.dump(res, f)
    fig = plot_convergence(res).get_figure()
    fig.savefig(f'{args_glb.output_dir}/output.png')
    fig2 = plot_evaluations(res)
    plt.savefig(f'{args_glb.output_dir}/output2.png')
    fig3 = plot_objective(res)
    plt.savefig(f'{args_glb.output_dir}/output3.png')

    return res



