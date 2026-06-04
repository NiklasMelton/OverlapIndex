"""
Repeated batch updates with partial_fit.

For offline backends, partial_fit refits/recomputes on the provided labeled
batch. For ARTMAP backends, partial_fit performs an incremental batch update.
"""

from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler
from overlapindex import OverlapIndex


def main() -> None:
    digits = load_digits()
    X = MinMaxScaler().fit_transform(digits.data)
    y = digits.target

    oi = OverlapIndex(
        model_type="Hypersphere",
        rho=0.85,
        match_tracking="MT+",
    )

    batch_size = 128
    for start in range(0, X.shape[0], batch_size):
        stop = start + batch_size
        oi.partial_fit(X[start:stop], y[start:stop])
        print(f"Processed {min(stop, X.shape[0]):4d} samples | OI = {oi.index:.6f}")


if __name__ == "__main__":
    main()
