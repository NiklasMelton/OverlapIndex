"""
Default offline OverlapIndex example on the Iris dataset.

This example uses the public sklearn-style API:
    oi.fit(X, y)
    score = oi.index

MiniBatchKMeans is the default backend.
"""

from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from overlapindex import OverlapIndex


def main() -> None:
    iris = load_iris()
    X = MinMaxScaler().fit_transform(iris.data)
    y = iris.target

    oi = OverlapIndex(
        kmeans_k=10,
        kmeans_kwargs={
            "random_state": 0,
            "batch_size": 32,
            "n_init": 1,
        },
    )

    oi.fit(X, y)

    print("Backend:", oi.model_type)
    print("Overlap Index:", oi.index)


if __name__ == "__main__":
    main()
