import os
from datetime import datetime
from radar import random_datetime
import random

from tqdm import tqdm


# 统计边界

def bound(path):
    longi_min = 100
    longi_max = -100
    lati_min = 100
    lati_max = 0
    with open(path) as f:
        time_min = '2013-01-01 00:36:00'
        time_max = '2013-06-30 22:41:32'

        for line in f.readlines():
            # time = datetime.strptime(line.split(',')[3], '%Y-%m-%d %H:%M:%S')
            longi = float(line.split(',')[4].split(' ')[1].replace('(', ''))
            if longi > longi_max:
                longi_max = longi
            if longi < longi_min:
                longi_min = longi

            lati = float(line.split(',')[4].split(' ')[2].replace(')', ''))
            if lati > lati_max:
                lati_max = lati
            if lati < lati_min:
                lati_min = lati
    return longi_min, longi_max, lati_min, lati_max


def generate_point(path, n):
    # 0,2013007001,30205037,2013-06-05 20:58:56,POINT (-73.993629 40.74176)
    longi_min, longi_max, lati_min, lati_max = bound('../resources/point_r')
    with open(path, 'w') as f:
        for i in tqdm(range(n)):
            prefix = '0,2013007001,30205037,'
            time = random_datetime('2013-01-01T00:36:00', '2013-06-30T22:41:32')
            longitude = random.uniform(longi_min, longi_max)

            latitude = random.uniform(lati_min, lati_max)
            line = f'{prefix}{str(time)},POINT ({round(longitude, 6)} {round(latitude, 6)})\n'
            f.write(line)



# generate_point('../resources/generate_100k/point_r', 100000)
# longi_min, longi_max, lati_min, lati_max = bound('../resources/point_r')
# print(longi_min, longi_max, lati_min, lati_max)
print(bound('../resources/point_r'))