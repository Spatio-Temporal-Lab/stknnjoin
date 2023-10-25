from utils.data import Data


def record(execute_time, args):
    data_type = args.data_type
    if data_type == 'point':
        record_point(execute_time, args)
    elif data_type == 'line':
        record_line(execute_time)
    elif data_type == 'polygon':
        record_polygon(execute_time)
    elif data_type == 'lp':
        record_lp(execute_time)

# 记录每一次执行的数据特征、参数以及对应执行时间，存储在历史仓库中(point)
def record_point(execute_time, args):
    target_file = f'{args.output_dir}/record'
    data_r = Data(args.data_file_r_local)
    data_s = Data(args.data_file_s_local)
    nums_r, timerange_r, longitude_range_r, latitude_range_r, sp_dist_r, tp_dist_r = data_r.data_info()
    nums_s, timerange_s, longitude_range_s, latitude_range_s, sp_dist_s, tp_dist_s = data_s.data_info()
    k = args.k
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
        f.write(f'{k}\t')

        for knob in ['alpha', 'beta', 'binNum']:
            f.write(f'{args.knobs_setting[knob]}\t')
        f.write(str(execute_time))
        f.write('\n')

# 记录每一次执行的数据特征、参数以及对应执行时间，存储在历史仓库中(line)
def record_line(execute_time, args):
    target_file = f'{args.output_dir}/record'
    data_r = Data(args.data_file_r_local)
    data_s = Data(args.data_file_s_local)
    nums_r, timerange_r, longitude_range_r, latitude_range_r, sp_dist_r, tp_dist_r, avg_point_r, distance_r = data_r.data_info_line()
    nums_s, timerange_s, longitude_range_s, latitude_range_s, sp_dist_s, tp_dist_s, avg_point_s, distance_s = data_s.data_info_line()
    k = args.k
    with open(f'{target_file}', 'a') as f:
        # 写入特征
        f.write(f'{nums_r}\t{timerange_r}\t{longitude_range_r}\t{latitude_range_r}\t{avg_point_r}\t{distance_r}\t')
        for sp_d_r in sp_dist_r:
            for x in sp_d_r:
                f.write(f'{int(x)}\t')
        for tp_r in tp_dist_r:
            f.write(f'{int(tp_r)}\t')
        f.write(f'{nums_s}\t{timerange_s}\t{longitude_range_s}\t{latitude_range_s}\t{avg_point_s}\t{distance_s}\t')
        for sp_d_s in sp_dist_s:
            for y in sp_d_s:
                f.write(f'{int(y)}\t')
        for tp_s in tp_dist_s:
            f.write(f'{int(tp_s)}\t')
        f.write(f'{k}\t')

        for knob in ['alpha', 'beta', 'binNum']:
            f.write(f'{args.knobs_setting[knob]}\t')
        f.write(str(execute_time))
        f.write('\n')

# 记录每一次执行的数据特征、参数以及对应执行时间，存储在历史仓库中(polygon)
def record_polygon(execute_time, args):
    target_file = f'{args.output_dir}/record'
    data_r = Data(args.data_file_r_local)
    data_s = Data(args.data_file_s_local)
    nums_r, timerange_r, longitude_range_r, latitude_range_r, sp_dist_r, tp_dist_r, avg_point_r, area_r = data_r.data_info_polygon()
    nums_s, timerange_s, longitude_range_s, latitude_range_s, sp_dist_s, tp_dist_s, avg_point_s, area_s = data_s.data_info_polygon()
    k = args.k
    with open(target_file, 'a') as f:
        # 写入特征
        f.write(f'{nums_r}\t{timerange_r}\t{longitude_range_r}\t{latitude_range_r}\t{avg_point_r}\t{area_r}\t')
        for sp_d_r in sp_dist_r:
            for x in sp_d_r:
                f.write(f'{int(x)}\t')
        for tp_r in tp_dist_r:
            f.write(f'{int(tp_r)}\t')
        f.write(f'{nums_s}\t{timerange_s}\t{longitude_range_s}\t{latitude_range_s}\t{avg_point_s}\t{area_s}\t')
        for sp_d_s in sp_dist_s:
            for y in sp_d_s:
                f.write(f'{int(y)}\t')
        for tp_s in tp_dist_s:
            f.write(f'{int(tp_s)}\t')
        f.write(f'{k}\t')

        for knob in ['alpha', 'beta', 'binNum']:
            f.write(f'{args.knobs_setting[knob]}\t')
        f.write(str(execute_time))
        f.write('\n')

# 记录每一次执行的数据特征、参数以及对应执行时间，存储在历史仓库中(lp)
def record_lp(execute_time, args):
    target_file = f'{args.output_dir}/record'
    data_r = Data(args.data_file_r_local)
    data_s = Data(args.data_file_s_local)
    nums_r, timerange_r, longitude_range_r, latitude_range_r, sp_dist_r, tp_dist_r, avg_point_r, area_r = data_r.data_info_line()
    nums_s, timerange_s, longitude_range_s, latitude_range_s, sp_dist_s, tp_dist_s, avg_point_s, area_s = data_s.data_info_polygon()
    k = args.k
    with open(target_file, 'a') as f:
        # 写入特征
        f.write(f'{nums_r}\t{timerange_r}\t{longitude_range_r}\t{latitude_range_r}\t{avg_point_r}\t{area_r}\t')
        for sp_d_r in sp_dist_r:
            for x in sp_d_r:
                f.write(f'{int(x)}\t')
        for st_r in tp_dist_r:
            f.write(f'{int(st_r)}\t')
        f.write(f'{nums_s}\t{timerange_s}\t{longitude_range_s}\t{latitude_range_s}\t{avg_point_s}\t{area_s}\t')
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
# def record_final_result(res, real_time, args):
#     if args.data_type == 'point':
#         for i, knob in enumerate(['alpha', 'beta', 'binNum']):
#             args.logger.print(f'{knob}: {res.x[i]}\t')
#         # args.logger.print(str(real_time))
#         record(real_time, 'data_model/point_w_tp')
#     elif args.data_type == 'line':
#         for i, knob in enumerate(['alpha', 'beta', 'binNum']):
#             args.logger.print(f'{knob}: {res.x[i]}\t')
#         # args.logger.print(str(real_time))
#         record_line(real_time, 'data_model/line_w_tp')
#     elif args.data_type == 'polygon':
#         for i, knob in enumerate(['alpha', 'beta', 'binNum']):
#             args.logger.print(f'{knob}: {res.x[i]}\t')
#         # args.logger.print(str(real_time))
#         record_polygon(real_time, 'data_model/polygon_w_tp')
#     elif args.data_type == 'lp':
#         for i, knob in enumerate(['alpha', 'beta', 'binNum']):
#             args.logger.print(f'{knob}: {res.x[i]}\t')
#         # args.logger.print(str(real_time))
#         record_lp(real_time, 'data_model/lp_w_tp')
