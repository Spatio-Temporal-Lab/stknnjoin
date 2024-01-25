from datetime import datetime

import pandas as pd
import re

s = '28db16cf3b709c2e45a4bf3ac1e3049d,c287bdbb06e25ee6858c3106db1b34b4,"[104.04226 30.69199 1541348099, 104.04256 30.69229 1541348102, 104.04256 30.69229 1541348105, 104.04333 30.69271 1541348109, 104.0442 30.69316 1541348115, 104.04463 30.69329 1541348118, 104.04503 30.69339 1541348121, 104.04571 30.69357 1541348126, 104.04584 30.69362 1541348128, 104.0461 30.69371 1541348130, 104.0469 30.69392 1541348134, 104.0472 30.694 1541348137, 104.04769 30.69409 1541348139, 104.04816 30.69417 1541348145, 104.0485 30.6942 1541348150, 104.04883 30.6942 1541348152, 104.04943 30.6942 1541348157, 104.04948 30.69421 1541348158, 104.04953 30.69425 1541348161, 104.04922 30.69439 1541348167, 104.04914 30.69438 1541348168, 104.04887 30.69433 1541348170, 104.04853 30.69432 1541348173, 104.0481 30.69435 1541348177, 104.04771 30.69433 1541348179, 104.04714 30.69431 1541348183, 104.04654 30.69423 1541348187, 104.0464 30.6942 1541348188, 104.04599 30.69408 1541348192, 104.04558 30.69391 1541348194, 104.04516 30.69376 1541348198, 104.04468 30.69356 1541348200, 104.04451 30.69346 1541348204, 104.04437 30.69332 1541348207, 104.04438 30.69332 1541348210, 104.04438 30.69323 1541348213, 104.04441 30.69312 1541348216, 104.04439 30.69311 1541348219, 104.04438 30.69311 1541348222, 104.04438 30.69311 1541348224, 104.04438 30.69311 1541348229, 104.04438 30.69311 1541348230, 104.04435 30.69302 1541348234, 104.0441 30.69279 1541348239, 104.0441 30.69279 1541348240, 104.04393 30.6926 1541348243, 104.04351 30.6927 1541348246, 104.04273 30.69271 1541348251, 104.04262 30.69261 1541348252, 104.04224 30.69237 1541348255]"'


def view(file):
    with open(file) as f:
        for i, line in enumerate(f):
            print(line)
            break


# def split_line(line):
#     line.strip()
#     id = line.split('"')[0]
#     traj = line.split('"')[1].replace('"', '')
#     traj = traj.replace('[', '').replace(']', '')
#     points_list = traj.split(', ')
#
#     for point in points_list:
#         longitude, latitude, time = point.split(', ')
#         time = datetime.fromtimestamp(int(time))
#
#     print(id)
#     print(traj)

# 从成都数据提取点文件
def get_point(file):
    file_r = open('../resources/chengdu/point_r', 'a')
    file_s = open('../resources/chengdu/point_s', 'a')
    with open(file) as f:
        for i, line in enumerate(f):
            line.strip()
            id = line.split('"')[0]
            traj = line.split('"')[1].replace('"', '')
            traj = traj.replace('[', '').replace(']', '')
            points_list = traj.split(', ')

            for point in points_list:
                longitude, latitude, time = point.split(' ')
                time = datetime.fromtimestamp(int(time))

                if i % 2 == 0:
                    # 0,2013010040,169716577,2013-04-24 06:33:58,POINT (-73.969513 40.800064)
                    file_r.write(f'0,0,0,{str(time)},POINT ({longitude} {latitude})\n')
                else:
                    file_s.write(f'0,0,0,{str(time)},POINT ({longitude} {latitude})\n')

    file_r.close()
    file_s.close()


# get_point('aa.csv')

# 从NY trip数据提取点
def save_as_point_file(file_path, point_r, point_s):
    point_r = open(point_r, 'a')
    point_s = open(point_s, 'a')

    def format_file(line):
        pickup_datetime = line[0]
        dropoff_datetime = line[1]
        pickup_longitude = float(line[2])
        if pickup_longitude > -73.75 or pickup_longitude < -74.07:
            return
        pickup_latitude = float(line[3])
        if pickup_latitude < 40.61 or pickup_latitude > 40.87:
            return
        dropoff_longitude = line[4]
        if dropoff_longitude > -73.75 or dropoff_longitude < -74.07:
            return
        dropoff_latitude = line[5]
        if dropoff_latitude < 40.61 or dropoff_latitude > 40.87:
            return
        line_r = f'0,0,0,{pickup_datetime},POINT ({pickup_longitude} {pickup_latitude})\n'
        point_r.write(line_r)
        line_s = f'0,0,0,{dropoff_datetime},POINT ({dropoff_longitude} {dropoff_latitude})\n'
        point_s.write(line_s)

    cols = [' pickup_datetime', ' dropoff_datetime', ' pickup_longitude', ' pickup_latitude', ' dropoff_longitude',
            ' dropoff_latitude']
    df = pd.read_csv(file_path)[cols]
    df.apply(format_file, axis='columns')

    point_r.close()
    point_s.close()


# 从didi西安提取线数据
def get_line(file, out):
    line_file = open(out, 'a')
    with open(file) as f:
        for i, line in enumerate(f):
            id = line.split('"')[0]
            traj = line.split('"')[1].replace('"', '')
            traj = traj.replace('[', '').replace(']', '')
            points_list = traj.split(', ')

            longitude_start, latitude_start, time_start = points_list[0].split(' ')
            time_start = datetime.fromtimestamp(int(time_start))
            content = f'LINESTRING ({longitude_start} {latitude_start}, '
            out_of_range = False
            for point in points_list[1:-1]:
                longitude, latitude, time = point.split(' ')
                if float(longitude) < 108.92 or float(longitude) > 109.01 or float(latitude) < 34.20 or float(latitude) > 34.28:
                    out_of_range = True
                    break
                # time = datetime.fromtimestamp(int(time))
                content = content + f'{longitude} {latitude}, '
            if not out_of_range:
                longitude_end, latitude_end, time_end = points_list[-1].split(' ')
                time_end = datetime.fromtimestamp(int(time_end))
                content = content + f'{longitude_end} {latitude_end})	{time_start}	{time_end}\n'
                line_file.write(content)
    line_file.close()



if __name__ == '__main__':
    # save_as_point_file('../data/new_york/trip_data_6.csv', '../data/point/point_r_6', '../data/point/point_s_6')
    get_line('../data/xian/xianshi_1115_1130.csv', '../data/line/line_1115_1130')
