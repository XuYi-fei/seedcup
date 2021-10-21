# baseline说明
1. 安装python（推荐miniconda），pytorch（参考官网教程）
2. 训练，运行train.py
3. 推理，运行test.py，生成提交文件（output_a.txt）
---
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