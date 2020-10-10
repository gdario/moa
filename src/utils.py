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


def create_image_arrays(df_list, remove_vehicles=False):
    assert len(df_list) == 2, "df_list must contain two data frames."
    coltran = ColumnTransformer([
        ('ohe', OneHotEncoder(), ['cp_dose']),
        ('drop_id', 'drop', ['sig_id', 'cp_type']),
    ], remainder=MinMaxScaler(feature_range=(0, 255)))

    if remove_vehicles:
        df_list = [df[df.cp_type == 'trt_cp'] for df in df_list]

    coltran.fit(df_list[0])
    transformed_df = [coltran.transform(df) for df in df_list]
    padded_df = [np.pad(df, ((0, 0), (12, 13)), 'constant',
                        constant_values=0).reshape(-1, 30, 30, 1)
                 for df in transformed_df]
    image_arrays = [np.repeat(x, 3, axis=3).astype('uint8') for x in padded_df]
    return image_arrays


def create_fname(df, img_folder):
    df['fname'] = img_folder + df.sig_id + '.png'
    return df


def save_images(x, dest_files):
    for i in range(len(dest_files)):
        data = Image.fromarray(x[i])
        data.save(dest_files[i])


def add_validation_flag(df, val_frac=0.2, seed=42):
    n_obs = df.shape[0]
    n_valid = int(n_obs*val_frac)
    np.random.seed(seed)
    idx = np.random.choice(np.arange(n_obs), size=n_valid, replace=False)
    df['is_valid'] = False
    df['is_valid'].iloc[idx] = True
    return df


def map_to_labels(targets, labels):
    out = []
    for i in range(targets.shape[0]):
        x = targets.iloc[i, 1:].values
        res = ' '.join(labels[x == 1].tolist())
        out.append(res)
    return out
