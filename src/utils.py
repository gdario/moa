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
