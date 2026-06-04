"""
Online ARTMAP OverlapIndex example on the Iris dataset.

Use an ARTMAP backend explicitly when you want single-sample or streaming
updates. This example demonstrates add_sample because it returns the current
Overlap Index after each update.
"""

from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from overlapindex import OverlapIndex


def main() -> None:
    iris = load_iris()
    X = MinMaxScaler().fit_transform(iris.data)
    y = iris.target

    oi = OverlapIndex(
        model_type="Hypersphere",
        rho=0.9,
        match_tracking="MT+",
    )

    score = oi.index
    for x_i, y_i in zip(X, y):
        score = oi.add_sample(x_i, int(y_i))

    print("Backend:", oi.model_type)
    print("Final Overlap Index:", score)
    print("Final Overlap Index from oi.index:", oi.index)


if __name__ == "__main__":
    main()
