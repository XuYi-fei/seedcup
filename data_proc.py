import random
import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame


def get_all_info() -> None:
    # 合并base_info和his_feature
    user_base_info = DataFrame(pd.read_csv(
        "./data/original/user_base_info.csv"))
    user_his_features = DataFrame(pd.read_csv(
        "./data/original/user_his_features.csv"))
    user_all_info = user_base_info.merge(
        user_his_features, how='right', on='id')
    user_all_info.to_csv("./data/original/user_all_info.csv")


def delete_columns() -> None:
    # 删除训练集和验证集中的无用列
    # x = ['real_age', 'utm_channel', 'add_all_num', 'view_all_num', 'msg_all_num']
    x = ['real_age', 'utm_channel']

    train = DataFrame(pd.read_csv("data/train/train.csv"))
    valid = DataFrame(pd.read_csv("data/train/valid.csv"))

    train.drop(columns=x, inplace=True)
    valid.drop(columns=x, inplace=True)

    train.to_csv("data/train/train.csv", sep=',', index=False, header=True)
    valid.to_csv("data/train/valid.csv", sep=',', index=False, header=True)


def add_track() -> None:
    # 把user_track信息填到总数据集中
    user_track = DataFrame(pd.read_csv("data/original/user_track.csv"))
    user_all_info = DataFrame(pd.read_csv(
        "data/original/user_all_info.csv"))
    work_day_count = [0] * (user_all_info.shape[0]+1)
    weekend_day_count = [0] * (user_all_info.shape[0]+1)
    avg_early_hour = [0.0] * (user_all_info.shape[0]+1)
    avg_last_hour = [0.0] * (user_all_info.shape[0]+1)
    total_day = [0] * (user_all_info.shape[0]+1)

    for i in range(user_track.shape[0]):
        # 统计登录日期类型
        total_day[user_track['id'][i]] += 1
        work_day_count[user_track['id'][i]] += user_track['is_weekend'][i] == 0
        weekend_day_count[user_track['id'][i]
                          ] += user_track['is_weekend'][i] != 0
        # 统计评价最早最晚登录时间
        avg_early_hour[user_track['id'][i]
                       ] = (avg_early_hour[user_track['id'][i]] * (total_day[user_track['id'][i]]-1) + user_track['early_hour'][i]) / total_day[user_track['id'][i]]
        avg_last_hour[user_track['id'][i]
                      ] = (avg_last_hour[user_track['id'][i]] * (total_day[user_track['id'][i]]-1) + user_track['last_hour'][i]) / total_day[user_track['id'][i]]

    # 保存到 all_info
    work_day_count.pop(0)
    weekend_day_count.pop(0)
    avg_early_hour.pop(0)
    avg_last_hour.pop(0)
    total_day.pop(0)
    user_all_info['total_day'] = total_day
    user_all_info['work_day_rate'] = [
        m/(n+0.001) for m, n in zip(work_day_count, total_day)]
    user_all_info['weekend_day_rate'] = [
        m/(n+0.001) for m, n in zip(weekend_day_count, total_day)]
    user_all_info['avg_early_hour'] = avg_early_hour
    user_all_info['avg_last_hour'] = avg_last_hour
    user_all_info.to_csv("data/original/all_info.csv")


def generate_test() -> None:
    test_a = DataFrame(pd.read_csv("data/original/test_a.csv"))
    all_info = DataFrame(pd.read_csv("data/train/all_info.csv"))

    result = all_info.merge(test_a, how='right', on='id')
    result.drop(columns=['label'], inplace=True)
    result.to_csv("data/test/test_info.csv", sep=',', index=False, header=True)


if __name__ == "__main__":
    # add_track()
    # delete_columns()
    generate_test()