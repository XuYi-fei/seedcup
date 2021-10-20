from sklearn import tree
import pandas as pd
import numpy as np
import graphviz
import time
import pickle
from config.tree_decision_config import config


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


def train():
    model = tree.DecisionTreeqClassifier(max_depth=config.max_depth, criterion=config.criterion,
                                        splitter=config.splitter)
    dataset = TreeTrainSeedDataset(config.train_data)
    model = model.fit(dataset.X, dataset.Y)

    # draw the tree, output file is named by current time
    dot_data = tree.export_graphviz(model, out_file=None)
    graph = graphviz.Source(dot_data)
    current_time = str(time.strftime("%m-%d-%H-%M", time.localtime()))
    graph.render(current_time)
    with open('treeCheckpoints\\' + current_time + '.pkl', 'wb') as output:
        pickle.dump(model, output)


def test():
    with open(config.model, 'rb') as input:
        model = pickle.load(input)
    dataset = TreeTestSeedDataset(config.test_data)
    result = model.predict(dataset.X)
    result = map(lambda x : str(x), result)
    with open("result.txt", "w") as f:
        f.write("\n".join(result))


if __name__ == "__main__":
    if config.test:
        test()
    else:
        train()
