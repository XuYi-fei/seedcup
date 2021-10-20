import pandas as pd
from pandas.core.frame import DataFrame


def get_all_info() -> None:
    user_base_info = DataFrame(pd.read_csv(
        "./data/original/user_base_info.csv"))
    user_his_features = DataFrame(pd.read_csv(
        "./data/original/user_his_features.csv"))
    user_all_info = user_base_info.merge(
        user_his_features, how='right', on='id')
    user_all_info.to_csv("./data/original/user_all_info.csv")


def delete_columns() -> None:
    # x = ['real_age', 'utm_channel', 'add_all_num', 'view_all_num', 'msg_all_num']
    x = ['real_age', 'utm_channel']

    test_a = DataFrame(pd.read_csv("data/v1/test_a.csv"))
    train = DataFrame(pd.read_csv("data/v1/train.csv"))
    valid = DataFrame(pd.read_csv("data/v1/valid.csv"))
    all_info = DataFrame(pd.read_csv("data/v2/all_info.csv"))
    test_a_info = DataFrame(pd.read_csv("data/v2/test_a_info.csv"))

    test_a.drop(columns=x, inplace=True)
    train.drop(columns=x, inplace=True)
    valid.drop(columns=x, inplace=True)
    all_info.drop(columns=x, inplace=True)
    test_a_info.drop(columns=x, inplace=True)

    test_a.to_csv("data/v1_p/test_a.csv", sep=',', index=False, header=True)
    train.to_csv("data/v1_p/train.csv", sep=',', index=False, header=True)
    valid.to_csv("data/v1_p/valid.csv", sep=',', index=False, header=True)
    all_info.to_csv("data/v2_p/all_info.csv",
                    sep=',', index=False, header=True)
    test_a_info.to_csv("data/v2_p/test_a_info.csv",
                       sep=',', index=False, header=True)


def add_track() -> None:
    user_track = DataFrame(pd.read_csv("data/original/user_track.csv"))
    work_day_count = [0] * 37045
    weekend_day_count = [0] * 37045
    avg_early_hour = [0.0] * 37045
    avg_last_hour = [0.0] * 37045
    count = 0
    last_id = 0

    for i in range(user_track.shape[0]):
        if(last_id != user_track['id'][i]):
            count = 0

        # 统计登录日期类型
        work_day_count[user_track['id'][i]] = user_track['is_weekend'][i] == 0
        weekend_day_count[user_track['id'][i]
                          ] = user_track['is_weekend'][i] == 0
        # 统计评价最早最晚登录时间
        count += 1
        avg_early_hour[user_track['id'][i]
                       ] = (avg_early_hour[user_track['id'][i]] * (count-1) + user_track['early_hour'][i]) / count
        avg_last_hour[user_track['id'][i]
                      ] = (avg_last_hour[user_track['id'][i]] * (count-1) + user_track['last_hour'][i]) / count

        last_id = user_track['id'][i]

    # 保存到 all_info
    user_all_info = DataFrame(pd.read_csv(
        "data/original/user_all_info.csv"))
    user_all_info['work_day_rate'] = [
        m/(m+n+0.001) for m, n in zip(work_day_count, weekend_day_count)]
    user_all_info['weekend_day_rate'] = [
        n/(m+n+0.001) for m, n in zip(work_day_count, weekend_day_count)]
    user_all_info['avg_early_hour'] = avg_early_hour
    user_all_info['avg_last_hour'] = avg_last_hour
    user_all_info.to_csv("data/original/user_all_info1.csv")


if __name__ == "__main__":
    add_track()
    # delete_columns()
