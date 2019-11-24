import os
import glob
import cv2
import numpy as np
import pandas as pd
from skimage import exposure
from sklearn.feature_extraction.image import extract_patches_2d

images_path = '../originals/'
labels_path = '../annotated-by-Anna/'

def load_csv(filename):
    csv_path = os.path.join(labels_path, filename)
    return pd.read_csv(csv_path, index_col=0)

def preprocess(x):
    return np.where(x < 128, 1, 0)

def balance_datasets(rois, labels):
    n = sum(labels)

    negative_idx = [i for i in range(len(labels)) if labels[i] == 0]
    np.random.shuffle(negative_idx)
    del_idx = negative_idx[np.int(n):]

    for idx in sorted(del_idx, reverse=True):
        del rois[idx]
        del labels[idx]

    assert len(rois) == 2 * n
    assert len(labels) == 2 * n

def get_patches_and_labels(filename, folder, patch_size, clip_limit):
    r = int((patch_size - 1)/2)

    regions = load_csv(filename)
    regions['patch_per_roi'] = (regions.height - patch_size + 1) * (regions.width - patch_size + 1)
    regions['bottom_px'] = regions.ycoord + regions.height
    regions['right_px'] = regions.xcoord + regions.width

    rois = []
    labels = []

    for label in regions.label.unique():

        img = cv2.imread(images_path + folder + label + ".tif")
        img = exposure.equalize_adapthist(img[:, :, 1], clip_limit=clip_limit) # keep only the green channel! 
        for _, row in regions[regions.label == label].iterrows():

            rois.extend(extract_patches_2d(img[row.ycoord : row.bottom_px, row.xcoord : row.right_px], 
                                           (patch_size, patch_size)))

            sklt = cv2.imread(labels_path + folder + label + '_' + str(row['sample']) + '_' + 'skeletone.jpg')
            if (sklt.ndim > 2):
                sklt = np.dot(sklt[...,:3], [0.299, 0.587, 0.114])
                sklt = sklt[r : - r, r : - r].reshape(-1, 1)
                sklt = preprocess(sklt)
            else:
                sklt = sklt[r : - r, r : - r].reshape(-1, 1)
            labels.extend(sklt)

    if sum(labels) / len(labels) < 0.3:
        balance_datasets(rois, labels)

    return np.asarray(rois), np.asarray(labels)

def get_patches(filename, folder, patch_size, clip_limit, name=""):
    r = int((patch_size - 1)/2)

    regions = load_csv(filename)
    regions['patch_per_roi'] = (regions.height - patch_size + 1) * (regions.width - patch_size + 1)
    regions['bottom_px'] = regions.ycoord + regions.height
    regions['right_px'] = regions.xcoord + regions.width

    if name:
        regions = regions[regions.label == name]

    rois = []

    for label in regions.label.unique():
        img = cv2.imread(images_path + folder + label + ".tif")
        img = exposure.equalize_adapthist(img[:, :, 1], clip_limit=clip_limit) # use only the green channel! 
        for _, row in regions[regions.label == label][0:3].iterrows():

            rois.extend(extract_patches_2d(img[row.ycoord : row.bottom_px, row.xcoord : row.right_px], 
                                           (patch_size, patch_size)))
    return np.asarray(rois)