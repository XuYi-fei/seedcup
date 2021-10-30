from sklearn import tree
import pandas as pd
import numpy as np
import graphviz
import time
import pickle
from config.tree_decision_config import config
from metric import *


class TreeTrainSeedDataset:

    def __init__(self, annotations_file):
        self.data: pd.DataFrame = pd.read_csv(annotations_file)
        self.data: pd.DataFrame = self.data[self.data['label'].notna()]
        self.Y = self.data['label']
        self.X = self.data.drop(['id', 'label'], axis=1).fillna(value=-1)
        self.X = np.array(self.X).tolist()
        self.Y = list(np.array(self.Y).astype(np.int64))

    def __len__(self):
        return len(self.data)


class TreeTestSeedDataset:
    def __init__(self, label_file):
        self.data: pd.DataFrame = pd.read_csv(label_file)
        self.X = self.data.drop(['id'], axis=1).fillna(value=-1)
        self.X = np.array(self.X).tolist()


def TrainModel() -> None:
    model = tree.DecisionTreeClassifier(max_depth=config.max_depth, criterion=config.criterion,
                                        splitter=config.splitter)
    dataset = TreeTrainSeedDataset(config.train_data)
    model = model.fit(dataset.X, dataset.Y)
    acc = ValidModel(model)
    # draw the tree, output file is named by current time
    dot_data = tree.export_graphviz(model, out_file=None)
    graph = graphviz.Source(dot_data, directory='./treeCheckpoints')
    current_time = str(time.strftime("%d-%H-%M-%S", time.localtime()))
    graph.render(current_time)
    with open('treeCheckpoints\\' + current_time + "-" + str(acc)[0:7] + '.pkl', 'wb') as output:
        pickle.dump(model, output)
        print("The model file is saved to " + 'treeCheckpoints\\' + current_time + "-" + str(acc)[0:5] + '.pkl')


def ValidModel(model) -> float:
    dataset = TreeTrainSeedDataset(config.valid_data)
    result = model.predict(dataset.X)
    total_num = len(result)
    acc = np.sum(np.array(result) - np.array(dataset.Y) == 0) / total_num
    P, R, F_score = precision_list(result, dataset.Y), recall_list(result, dataset.Y), f_score_list(result, dataset.Y)
    print("The valid accuracy ============>", acc, "%")
    print("The valid precision ============>", P, "%")
    print("The valid recall ============>", R, "%")
    print("The valid f_score ============>", F_score, "%")
    return acc


def TestModel():
    with open(config.model, 'rb') as input:
        model = pickle.load(input)
    dataset = TreeTestSeedDataset(config.test_data)
    result = model.predict(dataset.X)
    result = map(lambda x: str(x), result)
    with open("history/ML/DecisionTree_max-depth=3_0.8142.txt", "w") as f:
        f.write("\n".join(result))


if __name__ == "__main__":
    if config.test:
        TestModel()
    else:
        TrainModel()
