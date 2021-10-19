import pandas as pd
from pandas.core.frame import DataFrame


if __name__ == "__main__":
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
