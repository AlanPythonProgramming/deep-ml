import numpy as np

def gradient_boosting_step(X, y, current_predictions, learning_rate=0.1):
    X = np.asarray(X)
    preds = np.asarray(current_predictions, dtype=float)
    residuals = np.asarray(y) - preds

    best_mse = np.inf
    best_update = np.full(len(y), residuals.mean())

    for feature in X.T:
        values = np.unique(feature)
        thresholds = (values[:-1] + values[1:]) / 2

        for threshold in thresholds:
            left = feature <= threshold
            right = ~left

            left_mean = residuals[left].mean()
            right_mean = residuals[right].mean()

            update = np.where(left, left_mean, right_mean)
            mse = np.mean((residuals - update) ** 2)

            if mse < best_mse:
                best_mse = mse
                best_update = update

    return np.round(preds + learning_rate * best_update, 4).tolist()