import numpy as np
import os
import pandas as pd
from PIL import Image
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer


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


def create_image_arrays(df_list):
    """Convert a list of data frames into a list of image-like arrays."""
    assert len(df_list) == 2, "df_list must contain two data frames."
    coltran = ColumnTransformer([
        ('ohe', OneHotEncoder(), ['cp_dose']),
        ('drop_id', 'drop', ['sig_id', 'cp_type']),
    ], remainder=MinMaxScaler(feature_range=(0, 255)))
    coltran.fit(df_list[0])
    transformed_df = [coltran.transform(df) for df in df_list]
    padded_df = [np.pad(df, ((0, 0), (12, 13)), 'constant',
                        constant_values=0).reshape(-1, 30, 30, 1)
                 for df in transformed_df]
    image_arrays = [np.repeat(x, 3, axis=3).astype('uint8') for x in padded_df]
    return image_arrays


def create_fname(dataset):
    """Create the full path to an image file."""
    dataset['fname'] = dataset.sig_id + '.jpg'
    return dataset


def save_image(img, filename, folder):
    """Save an array as an image to a given filename."""
    data = Image.fromarray(img)
    data.save(os.path.join(folder, filename))


def add_validation_flag(dataset, val_frac=0.2, seed=42):
    """Add the 'is_valid' column."""
    n_obs = dataset.shape[0]
    n_valid = int(n_obs*val_frac)
    np.random.seed(seed)
    idx = np.random.choice(np.arange(n_obs), size=n_valid, replace=False)
    dataset['is_valid'] = False
    dataset['is_valid'].iloc[idx] = True
    return dataset


def map_to_lab(targets, labels):
    out = labels[targets == 1]
    if len(out) == 0:
        return 'none'
    else:
        return ' '.join(out.tolist())


def map_to_labels(targets, labels):
    X = targets.iloc[:, 1:].values
    out = []
    for i in range(X.shape[0]):
        out.append(map_to_lab(X[i], labels))
    return out
