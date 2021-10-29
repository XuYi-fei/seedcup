import pandas as pd
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

import argparse


class SVM_SVC:
    def __init__(self, clf, kernel, trainfile="../data/28_normalized/train.csv", validfile="../data/28_normalized/valid.csv", testfile="../data/28_normalized/test.csv"):
        super(SVM_SVC, self).__init__()

        train: pd.DataFrame = pd.read_csv(trainfile)
        train: pd.DataFrame = train[train['label'].notna()]
        valid: pd.DataFrame = pd.read_csv(validfile)
        valid: pd.DataFrame = valid[valid['label'].notna()]
        test: pd.DataFrame = pd.read_csv(testfile)

        self.train_y = train['label']
        self.train_x = train.drop(columns=['id', 'label']).fillna(value=-1)
        self.valid_y = valid['label']
        self.valid_x = valid.drop(columns=['id', 'label']).fillna(value=-1)
        self.test = test.drop(columns=['id']).fillna(value=-1)

        if(clf == "SVC"):
            if(kernel == "rbf"):
                self.classifier = Pipeline(
                    [("scaler", StandardScaler()), ("svm_clf", SVC(kernel="rbf", gamma=5, C=0.001))])
            if(kernel == "poly"):
                self.classifier = Pipeline([
                    ("scaler", StandardScaler()), ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=10))])

        if(clf == "LinearSVC"):
            self.classifier = Pipeline([("poly_featutres", PolynomialFeatures(degree=3)),
                                       ("scaler", StandardScaler()),  # 特征标准化
                                       ("svm_clf", LinearSVC(
                                           C=10, loss="hinge", random_state=42))  # 分类器
                                        ])

    def fit(self):
        self.classifier.fit(self.train_x, self.train_y)

    def score(self):
        return self.classifier.score(self.valid_x, self.valid_y)

    def predict(self):
        return self.classifier.predict(self.test)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--clf', type=str,
                        default="SVC")

    parser.add_argument('--kernel', type=str,
                        default="rbf")

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    svm = SVM_SVC(args.clf, args.kernel)
    svm.fit()
    print(svm.score())
    print(svm.predict())
