import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

X, y = make_moons(n_samples=100, noise=0.15, random_state=42)


def plot_dataset(X, y, axes):
    plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "bs")
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "g^")
    plt.axis(axes)
    plt.grid(True, which="both")
    plt.xlabel(r"$x_l$")
    plt.ylabel(r"$x_2$")


def plot_predict(clf, axes):
    x0s = np.linspace(axes[0], axes[1], 100)
    x1s = np.linspace(axes[2], axes[3], 100)
    x0, x1 = np.meshgrid(x0s, x1s)
    X = np.c_[x0.ravel(), x1.ravel()]
    print(f"{X.shape}\n\n")
    y_pred = clf.predict(X).reshape(x0.shape)
    y_decision = clf.decision_function(X).reshape(x0.shape)
    plt.contour(x0, x1, y_pred, cmap=plt.cm.winter, alpha=0.5)
    plt.contour(x0, x1, y_decision, cmap=plt.cm.winter, alpha=0.2)


if __name__ == "__main__":
    # LinearSVC
    polynomial_svm_clf = Pipeline([("poly_featutres", PolynomialFeatures(degree=3)),
                                   ("scaler", StandardScaler()),  # 特征标准化
                                   ("svm_clf", LinearSVC(
                                       C=10, loss="hinge", random_state=42))  # 分类器
                                   ])

    polynomial_svm_clf.fit(X, y)
    plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
    plot_predict(polynomial_svm_clf, [-1.5, 2.5, -1, 1.5])
    # plt.show()

    # SVC
    poly_kernel_svm_clf = Pipeline([("scaler", StandardScaler()),
                                    ("svm_clf", SVC(kernel="poly",
                                     degree=3, coef0=1, C=10))
                                    ])
    poly_kernel_svm_clf.fit(X, y)
    plt.subplot(132)
    plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
    plot_predict(poly_kernel_svm_clf, [-1.5, 2.5, -1, 1.5])

    # plt.show()

    #
    rbf_kernel_svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="rbf", gamma=5, C=0.001))
    ])

    plt.figure(figsize=(6, 3))

    plt.subplot(121)
    rbf_kernel_svm_clf.fit(X, y)
    plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
    plot_predict(rbf_kernel_svm_clf, [-1.5, 2.5, -1, 1.5])

    # plt.show()
