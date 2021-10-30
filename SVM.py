import pandas as pd
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

import argparse


class SVM_SVC:
    # def __init__(self, clf, kernel, C, degree, trainfile="data/ML/33_demension/train.csv", validfile="data/ML/33_demension/valid.csv", testfile="data/ML/33_demension/test.csv"):
    def __init__(self, clf, kernel, C, degree, trainfile="data/ML/28_demension/train.csv", validfile="data/ML/28_demension/valid.csv", testfile="data/ML/28_demension/test.csv"):
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
                self.classifier = Pipeline([
                    # ('pca',PCA()), 
                    ("scaler", StandardScaler()), 
                    ("svm_clf", SVC(kernel="rbf", gamma='auto', C=C))])
            if(kernel == "poly"):
                self.classifier = Pipeline([
                    # ('pca',PCA()), 
                    ("scaler", StandardScaler()), 
                    ("svm_clf", SVC(kernel="poly", degree=degree, coef0=1, C=C))])

        if(clf == "LinearSVC"):
            self.classifier = Pipeline([
                    ("poly_featutres", PolynomialFeatures(degree=degree)),
                    ("scaler", StandardScaler()),  # 特征标准化
                    ("svm_clf", LinearSVC(C=10, loss="hinge", random_state=42))])

    def fit(self):
        self.classifier.fit(self.train_x, self.train_y)

    def P(self):
        index_ = self.classifier.predict(self.valid_x) == 1
        TP = (self.valid_y[index_] == 1).sum()
        return round(TP / index_.sum(), 4)

    def R(self):
        index_ = self.valid_y == 1
        TP = (self.classifier.predict(self.valid_x)[index_] == 1).sum()

        return round(TP / index_.sum(), 4)
    
    def Fscore(self):
        P = self.P()
        R = self.R()

        return round(5 * P * R / (3 * P + 2 * R), 4)


    def predict(self):
        return self.classifier.predict(self.test).astype(float)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--clf', type=str,
                        default="SVC")

    parser.add_argument('--kernel', type=str,
                        default="rbf")
    parser.add_argument('--C', type=float,
                        default=1)
    parser.add_argument('--degree', type=int,
                        default=3)
    parser.add_argument('--auto', action='store_true',
                        help="auto predict by different hyperparameters")

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    if(args.auto == False):
        svm = SVM_SVC(args.clf, args.kernel, args.C, args.degree)
        svm.fit()
        print(f"valid:\tPrecision: {svm.P()}\tRecall: {svm.R()}\tFscore: {svm.Fscore()}")
        result = svm.predict().astype(int)
        # print(result)

        fp = open("output.txt", "w")
        for i in range(result.shape[0]):
            fp.write(result[i].astype(str))
            fp.write('\n')
    
    else:
        clf, kernel, degree = args.clf, args.kernel, args.degree
        C = 0
        resultfile = pd.DataFrame(columns=['C', 'Precision', 'Recall', 'Fscore'])


        for i in range(1, 500):
            C = float(i) / 10
            svm = SVM_SVC(clf, kernel, C, degree)
            svm.fit()
            P, R, Fscore = svm.P(), svm.R(), svm.Fscore()
            print(f"[C = {C}]\tPrecision: {P}\tRecall: {R}\tFscore: {Fscore}\n")
    
            resultfile.loc[i] = (str(C), str(P), str(R), str(Fscore))
        
        resultfile.to_csv("auto_result.csv")
        

