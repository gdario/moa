import utils
import numpy as np
from sklearn.compose import ColumnTransformer
# from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from utils import prepare_submission

import sys
sys.path.insert(0, '/home/giovenko/Projects/moa/src')

SEED = 42
NFOLDS = 5
np.random.seed(SEED)

# Load the data and the submission template
train, targets, test, sub = utils.load_data()

# Identify the rows containing the vehicle
idxv_tr = train.cp_type == 'ctl_vehicle'
idxv_te = test.cp_type == 'ctl_vehicle'

# We don't need the vehicles for training, but we still need the ids for the
# submission of the test set
train, targets = train[~idxv_tr], targets[~idxv_tr]

# ---------- Preprocessing ----------

y = targets.iloc[:, 1:].to_numpy()

coltrans = ColumnTransformer([
    ('drop_id', 'drop', ['sig_id', 'cp_type']),
    ('ordencode', OrdinalEncoder(), ['cp_dose'])
], remainder='passthrough')

# %%
pipe = Pipeline([
    ('preprocess', coltrans),
    ('scale', StandardScaler()),
    ('clf', OneVsRestClassifier(LogisticRegression(C=0.001)))
])

pipe.fit(train, y)

submission = prepare_submission(pipe, test, sub)
