import numpy as np
from typing import Tuple

def gini_impurity(y):
	return 1.0 - (y == 0).mean() ** 2 - (y == 1).mean() ** 2

def find_best_split(X: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
    """Return the (feature_index, threshold) that minimises weighted Gini impurity."""
    optimal_split = (None, None)
    min_gini = np.inf
    n = len(y)
    for feat_idx in range(X.shape[1]):
        values = np.unique(X[:, feat_idx])
        for th in values:
            left = y[X[:, feat_idx] <= th]
            right = y[X[:, feat_idx] > th]
            gini = (gini_impurity(left) * len(left) + gini_impurity(right) * len(right)) / n
            if min_gini > gini:
                min_gini = gini
                optimal_split = (feat_idx, th)
    return optimal_split 