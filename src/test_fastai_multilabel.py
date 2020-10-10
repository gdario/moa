import pandas as pd
from fastai.vision.all import *

df = pd.read_csv('../data/fastai_multilabel_df.csv')
# df.labels = df.labels.fillna('none')
dls = ImageDataLoaders.from_df(df, folder='../images/train',
                               valid_col='is_valid', label_delim=' ',
                               item_tfms=Resize(224))
learn = cnn_learner(dls, resnet50, metrics=partial(accuracy_multi, thresh=0.5))
learn.lr_find()
