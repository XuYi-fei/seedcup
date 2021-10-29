# 模型一：baseline
### 相关脚本

baseline_train.py、baseline_model.py、baseline_test.py

### 参数

`--evol` 	超参数自动优化

`--train、--valid、--in_feature` 	数据集路径和数据维度

### 目前效果

28维数据集（未归一化）——24轮——0.7065

&nbsp;

# 模型二：ResNet

### 说明

用ResNet网络预测

### 相关脚本

res_hyper-evol.py、res_model、res_test.py

### 参数

`--evol` 	超参数自动优化

`--device ` 	可选cpu、cuda

`--train、--valid、--in_feature` 	数据集路径和数据维度

### 目前效果

28维数据集（未归一化）——273轮——0.8196

&nbsp;

# 模型三：LCNet

### 说明

### 相关脚本

LCNet_train.py、LCNet_model.py、LCNet_test.py

### 参数

`--device ` 	可选cpu、cuda

`--train、--valid、--in_feature` 	数据集路径和数据维度

### 目前效果

33维数据集（未归一化）——131轮——0.8226

&nbsp;

# 模型四：SVM

### 说明

使用sklearn库调用SVM的现有模型进行预测，数据集为28维归一化后空值填-1的数据集

### 相关脚本

ML/SVM.py

### 数据集目录

`data/ML/`

### 参数

`--clf`	选择分类器：SVC、LinearSVC

`--kernel`	选择核函数：rbf、poly

`--C`	错误项的惩罚系数

### 目前效果

SVC—poly—degree=2—C=11—0.7656



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

# 模型结果的加权
### 1. `utils/vote.py`
- 脚本`utils/vote.py`
- 此脚本要求文件目录如下:
  
  ----history
  
  &nbsp;&nbsp;&nbsp;&nbsp;----model_name1

  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;----xxxxx_xxx_0.abcd.txt

  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;----xxxxx_xxx_0.abcd.pt

  &nbsp;&nbsp;&nbsp;&nbsp;----model_name2

  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;----xxxxx_xxx_0.efgh.txt

  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;----xxxxx_xxx_0.efgh.pt
  
- 注意不要更改`weighted`文件夹名

- 注意输出的txt结尾是四位小数的格式

&nbsp;

### 2. `test.py`

**直接对三种模型进行加权测试**
- 注意目前没有baseline的模型(主要是没有训好的baseline提交的分数)，如果有了之后，
  要和其他文件的命名方式一样并且放在history/Fake1DAttention下
- 模型的各种参数设置在`config/test_config.py`中，可自己查阅（目前baseline模型的config不正确，所以不要直接跑！！！！）
- 每次运行后，各个模型也会输出一个结果（在history下对应目录，反复运行会覆盖），总加权后的结果也会输出（也就是说目前是4个）

&nbsp;

# 数据集说明

### 当前模型对应数据集

| 模型     | train                  | valid                  | test                  |
| -------- | ---------------------- | ---------------------- | --------------------- |
| baseline | unmodified/train.csv   | unmodified/valid.csv   | unmodified/test_a.csv |
| ResNet   | 同上                   | 同上                   | 同上                  |
| LCNet    | 33_dimension/train.csv | 33_dimension/valid.csv | 33_dimension/test.csv |

### 数据分析

所有35维数据的相关性分析：

`/data/original/pearmon`、`/data/original/spearman` 分别保存了皮尔曼系数和斯皮尔曼系数

### 改动内容

从user_track中提取出总登录次数 total_day、工作日登录比例 work_day_rate、周末登录比例 weekend_day_rate、平均第一次登陆时间 avg_early_hour、平均最后一次登录时间avg_last_hour，现在完整数据见 data\original\all_info.csv

### 目录结构

<img src="C:\Users\lrk\AppData\Roaming\Typora\typora-user-images\image-20211023160734280.png" alt="image-20211023160734280" style="zoom:67%;" />

`33_dimension/` 	加入了user_track里的5个维度

`28_normalze/`	归一化后的28维数据集

`33_normalze/`	归一化后的33维数据集

`original/`	原始数据

`unmodified/`	baseline的数据集

### 随机切分数据

`utils/random_data.py` 用于随机切分数据，如果要在模型训练测试时使用,可以直接调用其中的`generate_new_data()`函数,生成的新数据
会在`data/random_data/`下

&nbsp;

# History目录

用于记录已提交文件测试效果及对应网络参数文件等

`/Fake1DAttention` 下为只使用全连接，文件命名格式：`数据维度(是否归一化)_训练轮数_测试分数`

`/ResNet` 下为使用残差网络，文件命名格式：`数据维度(是否归一化)_训练轮数_测试分数`

`/LCNet` 下为使用LCNet（conv），文件命名格式：`数据维度(是否归一化)_训练轮数_测试分数`

&nbsp;

# TODO

### lrk

###### 一期

​		数据预处理：transforms（已完成归一化）、删除部分维度（已完成相关性分析）

​		尝试不同：loss_fn、optimizer（均已完成，效果不佳）

​		超参数优化加到ResNet网络中（已完成），试试效果

###### 二期

​		尝试机器学习模型分类：LGBM、SVM（整理数据集ing）、随机森林

### xyf

### lc
