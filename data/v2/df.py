import pandas as pd
from pandas.core.frame import DataFrame


def check_exist(id, test_a) -> bool:
    for i in range(test_a.shape[0]):
        if(test_a['id'][i] == id):
            return True
    return False


if __name__ == "__main__":
    test_a = DataFrame(pd.read_csv("test_a.csv"))
    all_info = DataFrame(pd.read_csv("all_info.csv"))

    # method 1
    # for i in range(all_info.shape[0]):
    #     if check_exist(all_info['id'][i], test_a) != True:
    #         all_info = all_info.drop([i])
    # all_info.to_csv("./test_a_info.csv", sep=',', index=False, header=True)

    # method 2
    result = all_info.merge(test_a, how='right', on='id')
    result.drop(columns=['label'], inplace=True)
    result.to_csv("./test_a_info.csv", sep=',', index=False, header=True)

    # method 3
    # id_ = test_a['id']
    # result = all_info.query("id in @id_")
    # result.drop(columns=['label'], inplace=True)
    # result.to_csv("./test_a_info.csv", sep=',', index=False, header=True)
