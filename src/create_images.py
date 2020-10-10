import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer

train_data = pd.read_csv('../data/train_features.csv')
test_data = pd.read_csv('../data/test_features.csv')
train_targets = pd.read_csv('../data/train_targets_scored.csv')

idx_trt_train = train_data.cp_type == 'trt_cp'
idx_trt_test = test_data.cp_type == 'trt_cp'

train_novehicle = train_data.loc[idx_trt_train]
test_novehicle = test_data.loc[idx_trt_test]
targets_novehicle = train_targets.loc[idx_trt_train]

coltran = ColumnTransformer([
    ('ohe', OneHotEncoder(), ['cp_dose']),
    ('minmax', MinMaxScaler(), ['cp_time']),
    ('drop_id', 'drop', ['sig_id', 'cp_type']),

], remainder=StandardScaler())

coltran.fit(train_novehicle)
train_x = coltran.transform(train_novehicle)
test_x = coltran.transform(test_novehicle)

padded_x_train = np.pad(train_x, ((0, 0), (12, 13)), 'constant',
                        constant_values=0).reshape(-1, 30, 30, 1)
padded_x_test = np.pad(test_x, ((0, 0), (12, 13)), 'constant',
                       constant_values=0).reshape(-1, 30, 30, 1)

x_train = np.repeat(padded_x_train, 3, axis=3).astype('float32')
x_test = np.repeat(padded_x_test, 3, axis=3).astype('float32')
