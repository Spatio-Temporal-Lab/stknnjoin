def get_cols(data_type):
    """
    Parameters
    ----------
    data_type: str

    Returns
    -------
    cols: list
    """
    # print(data_type)
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
        cols.extend(['k', 'alpha', 'beta', 'binNum', 'execute_time'])

    elif data_type == 'line':
        cols = ['nums_r', 'timerange_r', 'longitude_range_r', 'latitude_range_r', 'avg_p_r', 'dist_r']
        for i in range(25):
            cols.append(f'sp_dist_r_{i + 1}')
        for i in range(30):
            cols.append(f'tp_dist_r_{i + 1}')
        cols.extend(['nums_s', 'timerange_s', 'longitude_range_s', 'latitude_range_s', 'avg_p_s', 'dist_s'])
        for i in range(25):
            cols.append(f'sp_dist_s_{i + 1}')
        for i in range(30):
            cols.append(f'tp_dist_s_{i + 1}')
        cols.extend(['k', 'alpha', 'beta', 'binNum', 'execute_time'])

    elif data_type == 'polygon':
        cols = ['nums_r', 'timerange_r', 'longitude_range_r', 'latitude_range_r', 'avg_p_r', 'area_r']
        for i in range(25):
            cols.append(f'sp_dist_r_{i + 1}')
        for i in range(30):
            cols.append(f'tp_dist_r_{i + 1}')
        cols.extend(['nums_s', 'timerange_s', 'longitude_range_s', 'latitude_range_s', 'avg_p_s', 'area_s'])
        for i in range(25):
            cols.append(f'sp_dist_s_{i + 1}')
        for i in range(30):
            cols.append(f'tp_dist_s_{i + 1}')
        cols.extend(['k', 'alpha', 'beta', 'binNum', 'execute_time'])

    elif data_type == 'lp':
        cols = ['nums_r', 'timerange_r', 'longitude_range_r', 'latitude_range_r', 'avg_p_r', 'dist_r']
        for i in range(25):
            cols.append(f'sp_dist_r_{i + 1}')
        for i in range(30):
            cols.append(f'tp_dist_r_{i + 1}')
        cols.extend(['nums_s', 'timerange_s', 'longitude_range_s', 'latitude_range_s', 'avg_p_s', 'area_s'])
        for i in range(25):
            cols.append(f'sp_dist_s_{i + 1}')
        for i in range(30):
            cols.append(f'tp_dist_s_{i + 1}')
        cols.extend(['k', 'alpha', 'beta', 'binNum', 'execute_time'])

    # print('len_cols', len(cols))
    try:
        print(len(cols))
        return cols
    except cols is None:
        print('get columns failed\r\n')


def get_features(data_type):
    if data_type == 'point':
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
    elif data_type == 'line':
        nums_r, timerange_r, longitude_range_r, latitude_range_r, sp_dist_r, tp_dist_r, avg_p_r, dist_r = data_r.data_info_line()
        nums_s, timerange_s, longitude_range_s, latitude_range_s, sp_dist_s, tp_dist_s, avg_p_s, dist_s = data_s.data_info_line()
        items = [nums_r, timerange_r, longitude_range_r, latitude_range_r, avg_p_r, dist_r]
        for dist in sp_dist_r:
            for x in dist:
                items.append(x)
        items.extend(tp_dist_r)
        items.extend([nums_s, timerange_s, longitude_range_s, latitude_range_s, avg_p_s, dist_s])
        for dist in sp_dist_s:
            for x in dist:
                items.append(x)
        items.extend(tp_dist_s)

    elif data_type == 'polygon':
        nums_r, timerange_r, longitude_range_r, latitude_range_r, sp_dist_r, tp_dist_r, avg_p_r, area_r = data_r.data_info_polygon()
        nums_s, timerange_s, longitude_range_s, latitude_range_s, sp_dist_s, tp_dist_s, avg_p_s, area_s = data_s.data_info_polygon()
        items = [nums_r, timerange_r, longitude_range_r, latitude_range_r, avg_p_r, area_r]
        for dist in sp_dist_r:
            for x in dist:
                items.append(x)
        items.extend(tp_dist_r)
        items.extend([nums_s, timerange_s, longitude_range_s, latitude_range_s, avg_p_s, area_s])
        for dist in sp_dist_s:
            for x in dist:
                items.append(x)
        items.extend(tp_dist_s)

    elif data_type == 'lp':
        nums_r, timerange_r, longitude_range_r, latitude_range_r, sp_dist_r, tp_dist_r, avg_p_r, distance_r = data_r.data_info_line()
        nums_s, timerange_s, longitude_range_s, latitude_range_s, sp_dist_s, tp_dist_s, avg_p_s, area_s = data_s.data_info_polygon()
        items = [nums_r, timerange_r, longitude_range_r, latitude_range_r, avg_p_r, distance_r]
        for dist in sp_dist_r:
            for x in dist:
                items.append(x)
        items.extend(tp_dist_r)
        items.extend([nums_s, timerange_s, longitude_range_s, latitude_range_s, avg_p_s, area_s])
        for dist in sp_dist_s:
            for x in dist:
                items.append(x)
        items.extend(tp_dist_s)

    k = global_args.get_value('k')
    items.append(k)
    items.extend([0, 0, 0])
    return items