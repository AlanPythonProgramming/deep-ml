import numpy as np
def random_forest_feature_importance(trees: list, n_features: int) -> list:
    """
    Calculate feature importance from a random forest using Mean Decrease in Impurity.
    
    Args:
        trees: List of trees, where each tree is a list of node splits.
               Each split is a dict with:
               - 'feature_index': int, the feature used for splitting
               - 'impurity_decrease': float, the weighted impurity decrease
        n_features: Total number of features in the dataset
    
    Returns:
        List of feature importances normalized to sum to 1.0
    """
    importances = np.zeros(n_features)
    has_split = False
    for tree in trees:
        for split in tree:
            has_split = True
            importances[split['feature_index']] += split['impurity_decrease']
    if has_split:
        return (importances / importances.sum()).tolist()
    return [0] * n_features
    
