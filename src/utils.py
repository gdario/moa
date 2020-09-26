import numpy as np


def fill_results(idx_cp, results):
    """Fill the result matrix

    Given a logical vector indicating the non-vehicle entries and an array of
    results, fill an array with as many rows as `idx_cp` and as many columns
    as `results` with the content of `results` leaving the rows corresponding
    to vehicles filled with zeros.
    """
    assert sum(idx_cp) == results.shape[0], "dimensions don't match"
    out = np.zeros((idx_cp.shape[0], results.shape[1]))
    out[idx_cp == 1] = results
    return out


def stabilize_y_hat(y_hat):
    y_high = np.zeros_like(y_hat)
    y_high.fill(1.0-1.0e-15)
    y_low = np.zeros_like(y_hat)
    y_low.fill(1.0e-15)
    return np.maximum(np.minimum(y_hat, y_high), y_low)


def log_loss_score(y_true, y_hat):
    y_hat = stabilize_y_hat(y_hat)
    out = (y_true*np.log(y_hat) + (1-y_true)*np.log(1-y_hat))
    return -out.mean(axis=0).mean()
