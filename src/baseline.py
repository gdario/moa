import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
import pandas as pd
import constants

train_ft = pd.read_csv(constants.DATA_DIR/'train_features.csv')
train_tgt_sc = pd.read_csv(constants.DATA_DIR/'train_targets_scored.csv')
test_ft = pd.read_csv(constants.DATA_DIR/'test_features.csv')

# There is no point in keeping the vehicle.
idx_trt_train = train_ft.cp_type == 'trt_cp'
idx_trt_test = test_ft.cp_type == 'trt_cp'

train_x = train_ft.loc[idx_trt_train]
# train_y = train_tgt_sc.loc[idx_trt_train]
train_y = train_tgt_sc.loc[idx_trt_train].iloc[:, 1:].to_numpy()
test_x = test_ft.loc[idx_trt_test]

col_trans = ColumnTransformer(
    [('encode', OrdinalEncoder(), ['cp_type', 'cp_dose']),
     ('drop_id', 'drop', 'sig_id')],
    remainder='passthrough'
)

clf = OneVsRestClassifier(LogisticRegression())
pipe = Pipeline([('preprocess', col_trans), ('clf', clf)])

n_fold = 0
train_losses = []
val_losses = []

ks = KFold(n_splits=5, shuffle=True, random_state=42)

for idx_train, idx_val in ks.split(train_x, train_y):
    print("Fold {}".format(n_fold))
    x_train, y_train = train_x.iloc[idx_train], train_y[idx_train]
    x_val, y_val = train_x.iloc[idx_val], train_y[idx_val]
    pipe.fit(x_train, y_train)
    ypred_train = pipe.predict_proba(x_train)
    ypred_val = pipe.predict_proba(x_val)
    loss_train = log_loss(y_train.ravel(), ypred_train.ravel())
    loss_val = log_loss(y_val.ravel(), ypred_val.ravel())
    train_losses.append(loss_train)
    val_losses.append(loss_val)
    n_fold += 1

# train_losses = np.array(train_losses)
# val_losses = np.array(val_losses)
