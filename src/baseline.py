import numpy as np
import pandas as pd
import constants
import utils
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.lin
from xgboost import XGBClassifier


train_ft = pd.read_csv(constants.DATA_DIR/'train_features.csv')
train_tgt_sc = pd.read_csv(constants.DATA_DIR/'train_targets_scored.csv')
test_ft = pd.read_csv(constants.DATA_DIR/'test_features.csv')

# There is no point in keeping the vehicle.
idx_veh_train = train_ft.cp_type == 'ctl_vehicle'
idx_veh_test = test_ft.cp_type == 'ctl_vehicle'

train_x = train_ft[~idx_veh_train]
train_y = train_tgt_sc[~idx_veh_train]
test = test_ft[~idx_veh_test]

# 1) Drop the id
# 2) Use OrdinalEncoder on 'cp_type', 'cp_dose'
column_trans = ColumnTransformer(
    [('cp_type', OrdinalEncoder(), ['cp_type', 'cp_dose']),
     ('sig_id', 'drop', 'sig_id')], remainder='passthrough'
)

x = column_trans.fit_transform(train_x)
y = train_y.iloc[:, 1:train_y.shape[1]].values

x_tr, x_val, y_tr, y_val = train_test_split(
    x, y, test_size=4000, random_state=42)

# # XGBoostClassifier
# params = {'max_depth': 2, 'eta': 1, 'objective': 'multi:softprob',
#           'eval_metric': 'mlogloss', 'num_class': 5, 'subsample': 0.8,
#           'colsample_bytree': 0.8}
# clf = OneVsRestClassifier(XGBClassifier(**params))
# clf.fit(x_tr[:1000], y_tr[:1000])
# pred_val = clf.predict(x_val)
# print(classification_report(y_val, pred_val))
