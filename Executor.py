import os
import pandas as pd
from tqdm import tqdm
import subprocess


class Executor:
    def __init__(self,
                 time_unit=30,
                 args=None):
        # 参数
        self.args = args
        self.alpha_list = [i for i in range(50, 1000, 100)]
        self.beta_list = [i for i in range(5, 100, 10)]
        self.binNum_list = [i for i in range(20, 700, 70)]
        self.k = args.k
        self.time_unit = time_unit

        # 数据路径
        self.r_file = args.data_file_r
        self.s_file = args.data_file_s

        # jar包
        self.jar_path = args.jar_path
        self.main_class = 'org.apache.spark.spatialjoin.app.SpatialJoinApp'

    # 解析结果
    def parse_result(self, file):
        with open(f'{file}/part-00000') as f:
            line = f.readline()
            return float(line.split(':')[1].strip())
    
    # hdfs读取结果
    def parse_result_hdfs(self, file):
        # out_byte = os.system(f'hdfs dfs -cat {file}/part-00000')
        out_byte = subprocess.check_output(f'hdfs dfs -cat {file}/part-00000 | head -1', shell=True)
        return float(str(out_byte, 'utf-8').split(':')[1].strip())

    # 执行
    def execute(self, alpha, beta, binNum, save_path):
        # java路径，本地windows下为 'java'
        # java_path = 'java'
        # java_path = '../java/jdk1.8.0_161/bin/java'
        java_path = self.args.java_path
        # 执行参数
        params = f'{alpha} {beta} {binNum} {self.time_unit} {self.k} {self.r_file} {self.s_file} {save_path}'
        # 执行指令
        # command = fr'{java_path} -cp {self.jar_path} {self.main_class} {params}'
        command = fr'spark-submit --conf spark.port.maxRetries=100 --num-executors 30 --executor-cores 4 --driver-memory 5g --conf "spark.default.parallelism=400" --executor-memory 8g --master yarn --class {self.main_class} {self.jar_path} {params}'
        print(command)
        # self.args.logger.print(f'command: {command}')
        os.system(command)

    def execute_specific_knob(self):
        # alpha 200, beta 40, binNum 200
        alpha = 200
        beta = 40
        binNum = 200

        bound_dict = {
            'alpha': [i for i in range(50, 1000, 10)],
            'beta': [i for i in range(5, 100, 1)],
            'binNum': [i for i in range(10, 1000, 10)]
        }
        self.args.logger.print(str(len(bound_dict[self.args.knob])))
        results_all = []
        for knob in bound_dict[self.args.knob]:
            if self.args.knob == 'alpha':
                alpha = knob
            elif self.args.knob == 'beta':
                beta = knob
            elif self.args.knob == 'binNum':
                binNum = knob

            time_sum = 0
            self.args.logger.print(f'alpha: {alpha}, beta: {beta}, binNum: {binNum}')
            cnt = 5
            for i in range(cnt):
                # alpha = 200, beta = 40, binNum = 200
                file_save_path = f'./results/generate/{alpha}_{beta}_{binNum}/{i}'
                if not os.path.exists(file_save_path):
                    self.execute(alpha=alpha, beta=beta, binNum=binNum, save_path=file_save_path)
                time = self.parse_result(file_save_path)
                self.args.logger.print(f'time: {time}')
                time_sum += time
            results_all.append({'alpha': alpha, 'beta': beta, 'binNum': binNum, 'time': time_sum / cnt})
            self.args.logger.print(f'avg_time: {time_sum / cnt}\n')

            pd.DataFrame(results_all).to_json(f'{self.args.save_path}/{self.args.knob}.json')
        return results_all

    def start(self):
        results_all = []
        for alpha in tqdm(self.alpha_list):
            for beta in self.beta_list:
                for binNum in self.binNum_list:

                    total_time = 0
                    for i in range(5):
                        file_save_path = f'./results/{alpha}_{beta}_{binNum}/{i}'
                        if os.path.exists(file_save_path):
                            continue
                        self.execute(alpha, beta, binNum, file_save_path)
                        total_time += self.parse_result(file_save_path)

                    avg_time = total_time / 5
                    self.args.logger.print(f'alpha: {alpha}, beta: {beta}, binNum: {binNum}, tome: {avg_time}')
                    results_all.append({'alpha': alpha, 'beta': beta, 'binNum': binNum, 'time': avg_time})

        df = pd.DataFrame(results_all)
        df.to_json(f'{self.args.save_path}/all.json')


if __name__ == '__main__':
    executor = Executor()
    executor.execute(alpha=200, beta=40, binNum=200, save_path='results_chengdu')
