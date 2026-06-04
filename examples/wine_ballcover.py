"""
BallCover OverlapIndex example on the Wine dataset.

The BallCover backend is offline-only. It builds one greedy landmark-ball cover
per class and exposes the balls as class-owned prototypes.
"""

from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler
from overlapindex import OverlapIndex


def main() -> None:
    wine = load_wine()
    X = MinMaxScaler().fit_transform(wine.data)
    y = wine.target

    oi = OverlapIndex(
        model_type="BallCover",
        ballcover_k=40,
        ballcover_radius="auto",
        ballcover_kwargs={
            "metric": "auto",
            "cover_fraction": 1.0,
            "random_state": 0,
        },
    )

    oi.fit(X, y)

    print("Backend:", oi.model_type)
    print("Overlap Index:", oi.index)

    # BallCover-specific diagnostics are available on the underlying backend.
    backend = oi._model
    if hasattr(backend, "resolved_metric"):
        print("Resolved metric:", backend.resolved_metric)
    if hasattr(backend, "n_clusters_total"):
        print("Total balls:", backend.n_clusters_total)


if __name__ == "__main__":
    main()
