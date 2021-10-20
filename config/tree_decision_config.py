import argparse


class TreeDecisionConfig:
    def __init__(self, configs):
        self.criterion = configs.criterion
        self.splitter = configs.splitter
        self.max_depth = configs.max_depth
        self.test = configs.test
        self.model = configs.model
        self.test_data = configs.test_data
        self.train_data = configs.train_data
        # self.min_samples_split = 2,
        # self.min_samples_leaf = 1,
        # self.min_weight_fraction_leaf = 0.,
        # self.max_features = None,
        # self.random_state = None,
        # self.max_leaf_nodes = None,
        # self.min_impurity_decrease = 0.,
        # self.min_impurity_split = None,
        # self.class_weight = None,
        # self.presort = 'deprecated',
        # self.ccp_alpha = 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-depth", help="max-depth of the decision tree", type=int, default=5)
    parser.add_argument("--criterion", "-c", help="judge criterion of the tree", type=str, choices=['gini', 'entropy'],
                        default='gini')
    parser.add_argument("--splitter", "-s", help="splitter strategy of the tree", type=str, choices=['best', 'random'],
                        default='best')
    parser.add_argument("--test", "-t", help="train or test", type=bool, default=True)
    parser.add_argument("--model", help="path to saved model", type=str, default='treeCheckpoints/10-20-14-18.pkl')
    parser.add_argument("--train-data", help="path to train data", type=str, default="./data/v1/train.csv")
    parser.add_argument("--test-data", help="path to test data", type=str, default="./data/v1/test_a.csv")
    return parser.parse_args()


arguments = parse_args()
config = TreeDecisionConfig(arguments)
