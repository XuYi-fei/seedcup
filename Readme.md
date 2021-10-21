# 2021.10.20

## 改动了数据集

从user_track中提取出总登录次数 total_day、工作日登录比例 work_day_rate、周末登录比例 weekend_day_rate、平均第一次登陆时间 avg_early_hour、平均最后一次登录时间avg_last_hour，现在完整数据见 data\original\all_info.csv



测试了昨天删除 `认证年龄、导流渠道`后40轮训练的效果，0.6336，太拉了



当前数据的目录结构：

<img src="https://gitee.com/lrk612/md_picture/raw/master/img/20211020194147.png" alt="image-20211020194140300" style="zoom:50%;" />

其中train下train为训练集、valid为验证集，test下为测试集

除去 id 和 label ，每条数据有33个维度

&nbsp;

&nbsp;

&nbsp;

# 2021.10.19

## 增加了超参数自动优化

##### 说明

用遗传算法在训练30轮后微调参数，进行下一个30轮的训练，总共演进100次，获得100组超参数（目前有：lr、positive_weight），取最优者来训练网络

##### 相关脚本

hyper_evol.py、train_modified.py

##### 使用方法

`python train_modified --evol`

&nbsp;

## 改动了数据集

##### 改动及效果

处理后的数据在 /data/v1_p、/data/v2_p目录下，历史记录：

1. 删除 `认证年龄、导流渠道、历史加好友数量、历史浏览好友数量、历史私聊好友数量` 

   效果：训练40轮，out.txt全是1

2. 删除 `认证年龄、导流渠道`

   效果：看着比 1. 好，明天试一试

##### 相关脚本

data_proc.py

&nbsp;

## TODO

##### 指标分析

对单个指标做方差（密集度）分析

对每个指标与 label 关联性做分析

指标间聚类分析、相关性分析

##### 训练集、数据集处理

user_track数据补充到总表

重选数据集，把 label=1 的都包含进来

空缺数据填充：平均值、插值、其他

##### 训练改进

loss_fn多采用几种

optimizer多采用几种
