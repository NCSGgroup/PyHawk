#!/bin/bash

### 设置该作业的作业名
#SBATCH --job-name=py-hawk

### 指定该作业需要2个节点数
#SBATCH --nodes=1

### 每个节点所运行的进程数为56
#SBATCH --ntasks-per-node=56

### 作业最大的运行时间，超过时间后作业资源会被SLURM回收
#SBATCH --time=15-00:00:00

### 提交到哪个分区
#SBATCH --partition=cpu-pgmf

### 错误输出文件
#SBATCH -e %j.%x.err

### 标准输出文件
#SBATCH -o %j.%x.out

##r 指定从哪个项目扣费。如果没有这条参数，则从个人账户扣费
#SBATCH --comment=public_cluster
### 程序的执行命令

source activate test5
cd /home/ud202180057/py-hawk-v2/demo/
python3 demo_1.py




