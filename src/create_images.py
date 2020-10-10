import os
import utils

SAVE_IMAGES = True

datafiles = ['../data/train_features.csv', '../data/test_features.csv']
datasets = [pd.read_csv(f) for f in datafiles]
train_targets = pd.read_csv('../data/train_targets_scored.csv')

idx_trt = [df.cp_type == 'trt_cp' for df in datasets]
sig_ids = [df.sig_id[df.cp_type == 'trt_cp'].values for df in datasets]

image_arrays = utils.create_image_arrays(datasets)

if SAVE_IMAGES:
    out_folders = ['../images/train', '../images/test']
    for i in range(len(image_arrays)):
        if not os.path.exists(out_folders[i]):
            os.makedirs(out_folders[i])
        print('Populating {}'.format(out_folders[i]))
        utils.save_images(image_arrays[i], sig_ids[i],
                          out_folders[i])