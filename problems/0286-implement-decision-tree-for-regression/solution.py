import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class Node:
    feat: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["Node"] = None
    right: Optional["Node"] = None
    value: Optional[float] = None

def split_node(X, y):
    optimal_split = (None, None)
    best_mse = np.inf
    n = len(y)
    for feat in range(X.shape[1]):
        candidates = np.unique(X[:, feat])
        for i in range(len(candidates) - 1):
            threshold = (candidates[i] + candidates[i+1]) / 2
            left = X[:, feat] <= threshold
            right = X[:, feat] > threshold
            left_mse = np.mean((y[left] - np.mean(y[left])) ** 2)
            right_mse = np.mean((y[right] - np.mean(y[right])) ** 2)

            weighted_mse = (
                np.sum(left) / n * left_mse
                + np.sum(right) / n * right_mse
            )
            if weighted_mse < best_mse:
                optimal_split = (feat, threshold)
                best_mse = weighted_mse
    return optimal_split

def build_tree(X, y, depth, max_depth, min_samples_split):
    # Check stopping conditions
    if depth >= max_depth or X.shape[0] < min_samples_split or (y == np.mean(y)).all():
        return Node(value=np.mean(y))

    # Find best feature and threshold
    feat, threshold = split_node(X, y)
    left = X[:, feat] <= threshold
    right = X[:, feat] > threshold
    
    # Split the dataset
    X_left, y_left = X[left, :], y[left]
    X_right, y_right = X[right, :], y[right]

    left = build_tree(X_left, y_left, depth + 1, max_depth, min_samples_split)
    right = build_tree(X_right, y_right, depth + 1, max_depth, min_samples_split)

    return Node(
        feat=feat,
        threshold=threshold,
        left=left,
        right=right
    )

def predict(X, node):
    if node.value is not None:
        return np.full(X.shape[0], node.value)
    preds = np.zeros(X.shape[0])
    left = X[:, node.feat] <= node.threshold
    right = X[:, node.feat] > node.threshold
    preds[left] = predict(X[left, :], node.left)
    preds[right] = predict(X[right, :], node.right)
    return preds

def decision_tree_regressor(X_train, y_train, X_test, max_depth=2, min_samples_split=2):
    """
    Build a decision tree for regression and predict on test data.
    
    Args:
        X_train: Training features, shape (n_samples, n_features)
        y_train: Training targets, shape (n_samples,)
        X_test: Test features, shape (m_samples, n_features)
        max_depth: Maximum depth of the tree
        min_samples_split: Minimum samples required to split a node
    
    Returns:
        List of predictions for X_test, rounded to 4 decimal places
    """
    X = np.array(X_train)
    y = np.array(y_train)
    tree = build_tree(X, y, 0, max_depth, min_samples_split)

    return predict(np.array(X_test), tree)
    

            

