from datetime import datetime
from geopy.distance import geodesic


class DataProcess:
    def __init__(self, from_path, to_path):
        self.from_path = from_path
        self.to_path = to_path

    def segment(self):
        """{'maxTimeIntervalInMinute': 2,
              'maxStayDistInMeter': 50,
              'minStayTimeInSecond': 30,
              'minTrajLengthInKM': 0.1,
              'segmenterType': 'HYBRID_SEGMENTER'
            }"""

        """{"maxStayDistInMeter":100,
                 |"minStayTimeInSecond":120,
                 |"stayPointType":"CLASSIC_DETECTOR"}"""
        # fbbd0a2bdea9b524963b5267371e1df6,b92f808142260c273ef6a16737ff6a50,"[108.95659 34.20502 1539040748, 108.95658 34.20521 1539040751, 108.95652 34.20536 1539040754, 108.9565 34.20557 1539040757, 108.95651 34.20588 1539040760, 108.95635 34.20603 1539040763, 108.95637 34.20622 1539040766, 108.95661 34.20635 1539040769, 108.9567 34.20655 1539040772, 108.95659 34.2068 1539040775, 108.95656 34.207 1539040778, 108.95654 34.20718 1539040781, 108.95641 34.20737 1539040784, 108.95651 34.20741 1539040787, 108.95658 34.20757 1539040790, 108.95647 34.20787 1539040793, 108.95646 34.20796 1539040796, 108.95639 34.20825 1539040799, 108.95645 34.20847 1539040802, 108.95643 34.20868 1539040805, 108.9564 34.20911 1539040808, 108.95659 34.20911 1539040811, 108.9567 34.20918 1539040814, 108.95653 34.20929 1539040817, 108.95654 34.20919 1539040820, 108.95662 34.20921 1539040823, 108.95659 34.20917 1539040826, 108.9566 34.20917 1539040829, 108.95663 34.20922 1539040832, 108.95663 34.20919 1539040835, 108.95662 34.20916 1539040838, 108.95662 34.20909 1539040841, 108.95661 34.20903 1539040844, 108.95638 34.20937 1539040847, 108.95642 34.20975 1539040850, 108.95631 34.21 1539040853, 108.95629 34.21006 1539040856, 108.95623 34.21017 1539040859, 108.95638 34.21043 1539040862, 108.95647 34.21055 1539040867, 108.95647 34.21066 1539040868, 108.95645 34.21074 1539040869]"
        f_line = open(self.to_path, 'a')
        with open(self.from_path) as f:
            for line in f:
                traj = line.split('[')[1].replace('"', '').replace(']', '')
                points_list = traj.split(', ')

                longitude_start, latitude_start, time_start = points_list[0].split(' ')
                time_start = datetime.fromtimestamp(int(time_start))

                lo_cur = float(longitude_start)
                la_cur = float(latitude_start)
                t_cur = time_start
                i_cur = 0
                st_point_all = []
                line_seg = []

                i = 0
                while i < len(points_list):
                    while True:
                        if i == len(points_list) - 1:
                            break
                        i += 1
                        point = points_list[i]
                        longitude, latitude, time = map(float, point.split(' '))
                        # time = datetime.fromtimestamp(int(time))
                        if geodesic((la_cur, lo_cur), (latitude, longitude))*1000 > 50:
                            i -= 1
                            break

                    if i_cur != i and (datetime.fromtimestamp(int(points_list[i].split(' ')[2]))-t_cur).seconds >= 30:
                        if st_point_all:
                            line_seg.append([st_point_all[-1][1]+1, i_cur-1])
                        else:
                            line_seg.append([0, i_cur-1])
                        st_point_all.append([i_cur, i])

                    if i == len(points_list) - 1:
                        break
                    i += 1
                    point = points_list[i]
                    longitude, latitude, time = map(float, point.split(' '))
                    time = datetime.fromtimestamp(int(time))
                    i_cur = i
                    lo_cur = longitude
                    la_cur = latitude
                    t_cur = time
                if not line_seg:
                    line_seg.append([0, len(points_list)])
                """{'maxTimeIntervalInMinute': 2,
                              'minTrajLengthInKM': 0.1,
                              'segmenterType': 'HYBRID_SEGMENTER'
                            }"""
                for seg in line_seg:

                    lo_s, la_s, t_s = map(float, point.split(' '))
                    t_s = datetime.fromtimestamp(int(t_s))
                    t = []
                    for point in points_list[seg[0]+1:seg[1]]:
                        longitude, latitude, time = map(float, point.split(' '))
                        time = datetime.fromtimestamp(int(time))
                        t.append([longitude, latitude])
                        if (time-t_s).seconds < 120 and geodesic((la_s, lo_s), (latitude, longitude)) > 0.1:
                            content = 'LINESTRING ('
                            for p in t:
                                content += f'{p[0]} {p[1]}, '
                            content = content[:-2] + f')	{t_s}	{time}\n'
                            f_line.write(content)
                            lo_s = longitude
                            la_s = latitude
                            t_s = time
                            t = []

        f_line.close()









if __name__ == '__main__':
    p = DataProcess('../data/xian/xianshi_1115_1130.csv', '../data/line/line_1115_1130_seg')
    p.segment()
