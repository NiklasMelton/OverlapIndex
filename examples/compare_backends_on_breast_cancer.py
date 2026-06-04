"""
Compare several OverlapIndex backends on the Breast Cancer Wisconsin dataset.

This example shows the same public fit/index API for offline backends and the
explicit ARTMAP selection path for online-capable backends.
"""

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from overlapindex import OverlapIndex


def main() -> None:
    data = load_breast_cancer()
    X = MinMaxScaler().fit_transform(data.data)
    y = data.target

    models = {
        "MiniBatchKMeans": OverlapIndex(
            model_type="MiniBatchKMeans",
            kmeans_k=12,
            kmeans_kwargs={
                "random_state": 0,
                "batch_size": 128,
                "n_init": 1,
            },
        ),
        "KMeans": OverlapIndex(
            model_type="KMeans",
            kmeans_k=12,
            kmeans_kwargs={"random_state": 0},
        ),
        "BallCover": OverlapIndex(
            model_type="BallCover",
            ballcover_k=48,
            ballcover_radius="auto",
            ballcover_kwargs={
                "metric": "auto",
                "cover_fraction": 1.0,
                "random_state": 0,
            },
        ),
        "Hypersphere": OverlapIndex(
            model_type="Hypersphere",
            rho=0.9,
            match_tracking="MT+",
        ),
    }

    for name, oi in models.items():
        oi.fit(X, y)
        print(f"{name:16s} OI = {oi.index:.6f}")


if __name__ == "__main__":
    main()
