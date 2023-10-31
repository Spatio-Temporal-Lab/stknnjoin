# Learning-Based Distributed Spatio-Temporal 𝑘 Nearest Neighbors Join

This repository contains the source code for the paper "Learning-Based Distributed Spatio-Temporal 𝑘 Nearest Neighbors Join". 

The code in this repository can be used to search for optimal paramerters for  ST-$k$NN Join with Gaussian -based BO and ML model. The output parameters can be effectively forwarded to the distributed ST-knn join program for actual execution. Interested users have the option to port the code to their own servers for execution and evaluation. 

## Source Code Structure

- `spatialjoin/`
- `stknn_knob_tuning/`
	- `config\`
	- `utils\`
	- `data_model.py`
	- `main.py`
	- `model.py`
	- `record.py`
	- `search.py`
	- `stknn_executor.py`

## Environment

All experiments are conducted on a cluster of 5 nodes, with each node equipped with CentOS 7.4, 24-core CPU and 128GB RAM. We deploy Hadoop 2.7.6 and Spark 2.3.3 in our cluster. During the experiments, we assign 5 cores and 5GB RAM to the driver program, and set up 30 executors in the Spark cluster. Each executor is assigned 5 cores and 16GB RAM.

Source code requires Python 3.7+ for running. 

## Run Experiments

Main scripts used for running experiments are `main.py`. These can be invoked using the following syntax:

```bash
python3 main.py <config_file_name>
```

where `<config_file_name>` is an experiment configuration file found in `config/*` directory. The format of the configuration file is as follows:
```
[file_section]  
data_file_r=/user/user1/stknn/data/line/line_r_5w_30d  
data_file_s=/user/user1/stknn/data/line/line_s_5w_30d  
data_file_r_local=data/line/line_r_5w_30d  
data_file_s_local=data/line/line_s_5w_30d  
save_dir=stknn/save/line/line_5w_5w_30d  
output_dir=log/line/line_5w_5w_30d  
  
[base_section]  
jar_path=jar/line_line.jar  
java_path=java  
knob=all  
k=15
```


### Usage Examples

```bash
python3 main.py default
```
