import numpy as np
import os
import pandas as pd
import utils

SAVE_IMAGES = True

datafiles = ['../data/train_features.csv', '../data/test_features.csv']
datasets = [pd.read_csv(f) for f in datafiles]
train_targets = pd.read_csv('../data/train_targets_scored.csv')

idx_trt = [df.cp_type == 'trt_cp' for df in datasets]
train_targets = train_targets.loc[idx_trt[0]]

datasets_trt = [datasets[i].loc[idx_trt[i]] for i in range(len(datasets))]
image_arrays = utils.create_image_arrays(datasets_trt)

img_folders = ['../images/train/', '../images/test/']
datasets_trt = [utils.create_fname(datasets_trt[i], img_folders[i])
                for i in range(len(datasets_trt))]

# sig_ids = [df.sig_id.values for df in datasets_trt]

if SAVE_IMAGES:
    for i in range(len(image_arrays)):
        if not os.path.exists(img_folders[i]):
            os.makedirs(img_folders[i])
        print('Populating {}'.format(img_folders[i]))
        for j in range(image_arrays[i].shape[0]):
            utils.save_image(image_arrays[i][j], datasets_trt[i].fname.iloc[j])

datasets_trt[0] = utils.add_validation_flag(datasets_trt[0])
datasets_trt[0]['labels'] = utils.map_to_labels(
    train_targets, np.array(train_targets.columns[1:].tolist()))

datasets_trt[0][['sig_id', 'fname', 'labels', 'is_valid']].to_csv(
    '../data/fastai_multilabel_df.csv', index=False)
