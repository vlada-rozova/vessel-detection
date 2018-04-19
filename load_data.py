import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

patch_size = 23
d = int((patch_size - 1)/2)

originals = np.zeros((885000, patch_size, patch_size), dtype = np.float32)
labels = np.zeros((885000, 1), dtype = np.float32)

originals_path = '/Users/vladarozova/Dropbox/PhD project/angiogenesis/samples for ridge detection'
labels_path = '/Users/vladarozova/Dropbox/PhD project/angiogenesis/Anna\'s results'
output = '/Users/vladarozova/Dropbox/PhD project/angiogenesis'

def load_csv(filename):
    csv_path = os.path.join(labels_path, filename)
    return pd.read_csv(csv_path)

def extract_patches(roi, height, width, length):
    patches = np.zeros((length, patch_size, patch_size), dtype = np.float32)
    patch_iter = 0
    for i in range(0, height - patch_size + 1):
        for j in range(0, width - patch_size + 1):
            patches[patch_iter, :, :] = roi[i : i + patch_size, j : j + patch_size]
            patch_iter += 1
    return patches

def process_rois(filename, folder, count):
    roi_data = load_csv(filename)
    originals_folder = os.path.join(originals_path, folder)
    labels_folder = os.path.join(labels_path, folder)

    N_patches = sum((roi_data['height'] - patch_size + 1) 
                    * (roi_data['width'] - patch_size + 1))
    rois = np.zeros((N_patches, patch_size, patch_size), dtype = np.float32)
    skeletons = np.zeros((N_patches, 1), dtype = np.float32)

    roi_iter = 0
    skelet_iter = 0
    for index, row in roi_data.iterrows():
        img = plt.imread(originals_folder + row['label'] + '.tif')
        height = row['height']
        width = row['width']
        patches_per_roi = (height - patch_size + 1) * (width - patch_size + 1)
        #length = patch_size * patches_per_roi
        length = patches_per_roi
#         rois[roi_iter : roi_iter + length, :, :] = extract_patches(img[row['ycoord'] :
#                                                                        row['ycoord'] + height, 
#                                                                        row['xcoord'] : 
#                                                                        row['xcoord'] + width, :],
#                                                                    height, width, length)
        rois[roi_iter : roi_iter + length, :, :] = extract_patches(img[row['ycoord'] :
                                                                       row['ycoord'] + height, 
                                                                       row['xcoord'] : 
                                                                       row['xcoord'] + width, 1],
                                                                   height, width, length)
        roi_iter += length
        
        img = plt.imread(labels_folder + row['label'] + '_' + str(row['sample']) + '_' + 'skeletone.jpg')
        img_grey = np.dot(img[...,:3], [0.299, 0.587, 0.114])
        skeletons[skelet_iter : skelet_iter + patches_per_roi] = img_grey[d : height - d, 
                                                                          d : width - d].reshape(-1, 1)

        skelet_iter += patches_per_roi
        
    labels[count : count + N_patches] = skeletons
    #originals[count * patch_size : count * patch_size + N_patches * patch_size, :, :] = rois
    originals[count : count + N_patches, :, :] = rois
    count += N_patches
    return originals, labels, count

