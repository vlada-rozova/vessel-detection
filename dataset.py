import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def binarize_labels(x, threshold, blurred = False):
    # Returns TRUE if it's a vessel, otherwise FALSE
    if blurred:
        x_binary = 0 * (x < threshold) + 1 * (x >= threshold)
    else:
        x_binary = 1 * (x < threshold) + 0 * (x >= threshold)
    return x_binary

def standardize(x, mean, sd):
    x_std = (x - mean) / sd
    print('The set has mean', np.mean(x), 'and s.d.', np.std(x, ddof = 1))
    print('Standardized set has mean', np.mean(x_std), 'and s.d.', np.std(x_std, ddof = 1))
    return x_std