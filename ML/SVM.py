import pandas as pd
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures


class SVM_SVC:
    def __init__(self, trainfile, testfile):
        super(SVM_SVC, self).__init__()

        data: pd.DataFrame = pd.read_csv(trainfile)
        data: pd.DataFrame = data[data['label'].notna()]
        test: pd.DataFrame = pd.read_csv(testfile)
        self.label = data['label']
        self.train = data.drop(columns=['id', 'label']).fillna(value=-1)
        self.test = test.drop(columns=['id']).fillna(value=-1)

        self.classifier = Pipeline([
            ("scaler", StandardScaler()),
            ("svm_clf", SVC(kernel="rbf", gamma=5, C=0.001))
        ])

    def fit(self):
        self.classifier.fit(self.train, self.label)

    def predict(self):
        return self.classifier.predict(self.test)


if __name__ == "__main__":

    svm = SVM_SVC("../data/unmodified/train.csv",
                  "../data/unmodified/test_a.csv")
    svm.fit()
    print(svm.predict())
