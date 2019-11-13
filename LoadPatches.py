import os
import pandas as pd
import numpy as np
import cv2
from skimage import exposure
import matplotlib.pyplot as plt

patch_size = 23
d = int((patch_size - 1)/2)

originals = np.zeros((885000, patch_size, patch_size), dtype = np.float32)
labels = np.zeros((885000, 1), dtype = np.float32)

originals_path = '/Users/vladarozova/Dropbox/PhD/angiogenesis/originals'
labels_path = '/Users/vladarozova/Dropbox/PhD/angiogenesis/Anna\'s results'

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

def extract_patches_2(roi, patches_rows, patches_cols, patches_per_roi):
    patches = np.zeros((patches_per_roi, patch_size, patch_size), dtype = np.float32)
    patch_iter = 0
    
    for i in range(0, patches_rows):
        for j in range (0, patches_cols):
            patches[patch_iter, :, :] = roi[i * patch_size : (i + 1) * patch_size,
                                            j * patch_size : (j + 1) * patch_size]
            patch_iter += 1
    return patches

def process_rois(filename, folder, count, patch_size=23, clip_limit=0.03):
    d = int((patch_size - 1)/2)
    
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
        img = cv2.imread(originals_folder + row['label'] + '.tif')
        img = exposure.equalize_adapthist(img, clip_limit=clip_limit)
        height = row['height']
        width = row['width']
        patches_per_roi = (height - patch_size + 1) * (width - patch_size + 1)
        length = patches_per_roi
        rois[roi_iter : roi_iter + length, :, :] = extract_patches(img[row['ycoord'] :
                                                                       row['ycoord'] + height, 
                                                                       row['xcoord'] : 
                                                                       row['xcoord'] + width, 1],
                                                                   height, width, length)
        roi_iter += length
        
        img = cv2.imread(labels_folder + row['label'] + '_' + str(row['sample']) + '_' + 'skeletone.jpg')
        if (img.ndim > 2):
            img_grey = np.dot(img[...,:3], [0.299, 0.587, 0.114])
            skeletons[skelet_iter : skelet_iter + patches_per_roi] = img_grey[d : height - d, 
                                                                          d : width - d].reshape(-1, 1)
        else:
            skeletons[skelet_iter : skelet_iter + patches_per_roi] = img[d : height - d, 
                                                                          d : width - d].reshape(-1, 1)
            
        skelet_iter += patches_per_roi
        
    labels[count : count + N_patches] = skeletons
    originals[count : count + N_patches, :, :] = rois
    count += N_patches
    return originals, labels, count

def process_rois_2(filename, folder, count):
    roi_data = load_csv(filename)
    originals_folder = os.path.join(originals_path, folder)
    labels_folder = os.path.join(labels_path, folder)
    
    N_patches = int(sum(np.floor(roi_data['height'] / patch_size) *
                    np.floor(roi_data['width'] / patch_size)))

    rois = np.zeros((N_patches, patch_size, patch_size), dtype = np.float32)
    skeletons = np.zeros((N_patches, 1), dtype = np.float32)

    roi_iter = 0
    skelet_iter = 0
    for index, row in roi_data.iterrows():
        img = plt.imread(originals_folder + row['label'] + '.tif')
        height = row['height']
        width = row['width']
        patches_rows = int(np.floor(height / patch_size))
        patches_cols = int(np.floor(width / patch_size))
        patches_per_roi = patches_rows * patches_cols
        #length = patches_per_roi
        
        rois[roi_iter : roi_iter + patches_per_roi, :, :] = extract_patches_2(img[row['ycoord'] :
                                                                                  row['ycoord'] + height, 
                                                                                  row['xcoord'] : 
                                                                                  row['xcoord'] + width, 1],
                                                                              patches_rows, patches_cols, patches_per_roi)
        roi_iter += patches_per_roi
        
        #img = plt.imread(labels_folder + row['label'] + '_' + str(row['sample']) + '_' + 'skeletone.jpg')
        #if (img.ndim > 2):
        #    img_grey = np.dot(img[...,:3], [0.299, 0.587, 0.114])
        #    skeletons[skelet_iter : skelet_iter + patches_per_roi] = img_grey[d : height - d, 
        #                                                                  d : width - d].reshape(-1, 1)
        #else:
        #    skeletons[skelet_iter : skelet_iter + patches_per_roi] = img[d : height - d, 
        #                                                                  d : width - d].reshape(-1, 1)
        #    
        #skelet_iter += patches_per_roi
        
    labels[count : count + N_patches] = skeletons
    originals[count : count + N_patches, :, :] = rois
    count += N_patches
    return originals, labels, count

def plot_patches(x, height, width, patch_size):
    nrows, rem_i = divmod(height, patch_size)
    ncols, rem_j = divmod(width, patch_size)
    roi_height = height - patch_size + 1
    roi_width = width - patch_size + 1
    x_ = np.zeros((height, width), dtype = np.float32)
    for i in range(0, nrows):
        for j in range(0, ncols):
            x_[i * patch_size : (i + 1) * patch_size, 
               j * patch_size : (j + 1) * patch_size] = \
            x[(i * roi_width + j) * patch_size, :, :]                       
            if (j == ncols - 1):
                x_[i * patch_size : (i + 1) * patch_size, -patch_size :] = \
                x[(i * roi_width + j) * patch_size + rem_j, :, :]
            if (i == nrows - 1):
                x_[-patch_size :, j * patch_size : (j + 1) * patch_size] = \
                x[(i * roi_width + j) * patch_size + rem_i * roi_width, :, :]
    x_[- patch_size:, -patch_size:] = x[-1, :, :]
    
    plt.figure(figsize=(5, 5))
    plt.imshow(x_, cmap = plt.cm.gray, interpolation = "nearest")
    plt.show()

def plot_skeleton(x, height, width, patch_size):
    roi_height = height - patch_size + 1
    roi_width = width - patch_size + 1
    x_ = x.reshape(roi_height, roi_width)
    
    plt.figure(figsize=(5, 5))
    plt.imshow(x_, cmap = plt.cm.gray, interpolation = "nearest")
    plt.show()
    
def plot_patches_without_overlap(x, nrows, ncols, patch_size):
    x_ = np.zeros((nrows * patch_size, ncols * patch_size), dtype = np.float32)
    for i in range(0, nrows):
        for j in range(0, ncols):
            x_[i * patch_size : (i + 1) * patch_size, 
               j * patch_size : (j + 1) * patch_size] = x[i * ncols + j, :, :]

    plt.figure(figsize=(10, 10))
    plt.imshow(x_, cmap = plt.cm.gray, interpolation = "nearest")
    plt.show()
