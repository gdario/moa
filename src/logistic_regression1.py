# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import utils
from constants import DATA_DIR
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_validate, KFold
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

import sys
sys.path.insert(0, '/home/giovenko/Projects/moa/src')

SEED = 42
NFOLDS = 5
np.random.seed(SEED)

# %%
train = pd.read_csv(DATA_DIR/'train_features.csv')
targets = pd.read_csv(DATA_DIR/'train_targets_scored.csv')

test = pd.read_csv(DATA_DIR/'test_features.csv')
sub = pd.read_csv(DATA_DIR/'sample_submission.csv')

y = targets.iloc[:, 1:].to_numpy()

# %%
coltrans = ColumnTransformer([
    ('drop_id', 'drop', 'sig_id'),
    ('ordencode', OrdinalEncoder(), ['cp_type', 'cp_dose'])
], remainder='passthrough')

# %%
pipe = Pipeline([
    ('preprocessing', coltrans),
    ('clf', OneVsRestClassifier(LogisticRegression()))
])

# %%
train_losses = []
val_losses = []
n_fold = 0
kfcv = KFold(n_splits=5, shuffle=False)
for idx_train, idx_val in kfcv.split(train, y):
    print('Fold {}'.format(n_fold))
    y_train, y_val = y[idx_train], y[idx_val]
    pipe.fit(train.iloc[idx_train], y_train)
    p_train = pipe.predict_proba(train.iloc[idx_train])
    p_val = pipe.predict_proba(train.iloc[idx_val])
    loss_train = log_loss(y_train.ravel(), p_train.ravel())
    loss_val = log_loss(y_val.ravel(), p_val.ravel())
    train_losses.append(loss_train)
    val_losses.append(loss_val)
    n_fold += 1

# %%
