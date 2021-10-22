# baseline说明
### 训练 train_hyper-evol.py

——train 训练集路径

——valid 验证集路径

——in_feature 训练集维度

——evol 自动超参数优化

### 测试 test.py

&nbsp;

# 决策树说明
### 代码说明

- ``decision_tree.py``: 运行决策树train和test的文件
- ``config/tree_decistion_config.py``: 决策树的相关配置
- train后会生成一个pdf和一个模型文件，pdf是决策树的可视化pdf;模型文件保存在treeCheckpoints下
- test时需要指定具体的模型文件
### 运行说明

- 具体配置参见`config/tree_decision_config.py`
- 训练代码示例:
    ```bash
    python decision_tree.py
  ```
- 测试代码示例:
    ```bash
    python decision_tree.py --test True --model treeCheckpoints/10-20-14-18.pkl
    ```

&nbsp;

# 数据集说明

### 数据分析

所有35维数据的相关性分析：

`/data/original/pearmon`、`/data/original/spearman` 分别保存了皮尔曼系数和斯皮尔曼系数

### 改动内容

从user_track中提取出总登录次数 total_day、工作日登录比例 work_day_rate、周末登录比例 weekend_day_rate、平均第一次登陆时间 avg_early_hour、平均最后一次登录时间avg_last_hour，现在完整数据见 data\original\all_info.csv

### 使用说明

当前数据的目录结构：

<img src="https://gitee.com/lrk612/md_picture/raw/master/img/20211021173218.png" alt="image-20211020194140300" style="zoom:50%;" />

其中train下train为训练集、valid为验证集，test下为测试集

除去 id 和 label ，每条数据有33个维度

&nbsp;

# 超参数自动优化

### 说明

用遗传算法在训练30轮后微调参数，进行下一个30轮的训练，总共演进100次（可选），获得100组超参数（目前有：lr、positive_weight），取最优者来训练网络

### 相关脚本

hyper_evol.py、train_hyper-evol.py

### 使用方法

`python train_hyper-evol --evol`

&nbsp;

# History目录

用于记录已提交文件测试效果及对应网络参数文件等

`/Fake1DAttention` 下为只使用全连接，文件命名格式：`数据维度_训练轮数_测试分数`

`/ResNet` 下为使用残差网络，文件命名格式：`数据维度_训练轮数_测试分数`

&nbsp;

# TODO

### lrk

归档本地文件，添加到main

数据预处理：transforms、删除部分维度

尝试不同：loss_fn、optimizer

超参数优化加到ResNet网络中，试试效果

### xyf

### lc
