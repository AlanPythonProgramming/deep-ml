import numpy as np

def find_best_histogram_split(gradient_hist, hessian_hist, lambda_: float = 1.0, gamma: float = 0.0) -> dict:
    """
    Find the best split across feature histograms in gradient boosting.

    Args:
        gradient_hist: 2D array (n_features, n_bins) of summed gradients per bin
        hessian_hist:  2D array (n_features, n_bins) of summed hessians per bin
        lambda_: L2 regularization on leaf weights
        gamma:   minimum gain threshold (penalty per split)

    Returns:
        dict with keys 'feature', 'bin', 'gain'
    """
    gradient_hist = np.array(gradient_hist)
    hessian_hist = np.array(hessian_hist)
    feat, b = -1, -1
    max_gain = 0.0
    for feat_num in range(gradient_hist.shape[0]):
        for b_num in range(gradient_hist.shape[1]-1):
            hl = np.sum(hessian_hist[feat_num, :b_num+1])
            hr = np.sum(hessian_hist[feat_num, b_num+1:])
            if hl > 0 and hr > 0:
                gl = np.sum(gradient_hist[feat_num, :b_num+1])
                gr = np.sum(gradient_hist[feat_num, b_num+1:])
                gain = 0.5 * (gl**2/(hl+lambda_) + gr**2/(hr+lambda_) - (gl+gr)**2/(hl+hr+lambda_)) - gamma
                if gain > max_gain:
                    max_gain = gain
                    feat = feat_num
                    b = b_num
    return {
        "feature": round(feat, 4),
        "bin": round(b, 4),
        "gain": round(max_gain, 4)
    }