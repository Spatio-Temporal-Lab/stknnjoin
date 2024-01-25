import os
import subprocess
from hdfs import InsecureClient

# %%
# 连接到HDFS
# hdfs_client = InsecureClient('http://your-hdfs-host:your-hdfs-port', user='your-hadoop-username')
hdfs_client = InsecureClient('http://10.242.6.16:50070', user='user1')

# %%
# 指定要读取的HDFS目录
hdfs_directory = 'stknn/save/node/pt_pt_300'

# 获取目录下所有子目录
subdirectories = [subdir for subdir in hdfs_client.list(hdfs_directory) if hdfs_client.status(f"{hdfs_directory}/{subdir}")['type'] == 'DIRECTORY']
print(len(subdirectories))

# 遍历每个part-0000文件，读取内容并保存到本地文件
local_file = open('../save/node_pt_pt/5.csv', 'a')
content = ""
for subdir in subdirectories:
    hdfs_file_path = f"{hdfs_directory}/{subdir}/0/part-00000"

    # 获取目录名并使用下划线进行分割
    split_result = subdir.split('_')  # 使用下划线分割目录名
    with hdfs_client.read(hdfs_file_path) as hdfs_file:
         res = hdfs_file.readline().decode('utf-8')
         content += f"pt_pt, 5, {split_result[0]}, {split_result[1]}, {split_result[2]}, {res[10:]}"
local_file.write(content)

local_file.close()

# out_byte = subprocess.check_output(f'hdfs dfs -cat {file}/part-00000 | head -1', shell=True)
