from matplotlib import pyplot as plt
import numpy as np
import cv2

import pandas as pd
import seaborn as sn
from skimage.feature import local_binary_pattern as LBP
import tensorflow as tf

def localSD(mat, n):
    mat = np.float32(mat)
    mu = cv2.blur(mat, (n, n))
    mdiff = mu - mat
    mat2 = cv2.blur(np.float64(mdiff * mdiff), (n, n))
    sd = np.float32(cv2.sqrt(mat2))

    return sd

def get_features(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    l, a, bb = cv2.split(lab)
    h, s, v = cv2.split(hsv)
    b, g, r = cv2.split(bgr)
    chan_funs = []
    img_size = b.shape

    ddepth = cv2.CV_16S

    for c in [b, g, r, h, s, v, l, a, bb]:
        chan_funs.append(c)
        chan_funs.append(cv2.GaussianBlur(c, (15, 15), cv2.BORDER_DEFAULT))
        chan_funs.append(cv2.GaussianBlur(c, (31, 31), cv2.BORDER_DEFAULT))
        chan_funs.append(localSD(c, 127))
        chan_funs.append(localSD(c, 63))
        chan_funs.append(localSD(c, 31))
        chan_funs.append(LBP(c, 32, 4, method="ror"))
        chan_funs.append(LBP(c, 24, 3, method="ror"))
        chan_funs.append(LBP(c, 16, 2, method="ror"))
        chan_funs.append(cv2.Laplacian(c, ddepth, ksize=3))
        chan_funs.append(cv2.Laplacian(c, ddepth, ksize=7))
        chan_funs.append(cv2.Laplacian(c, ddepth, ksize=15))
    ravels = []
    for cf in chan_funs:
        ravels.append(cf.ravel().T)
    feat = np.vstack(ravels).T
    return feat

def confusion(test_labels, predicted_classes):
    # Compute confusion matrix
    cm = tf.math.confusion_matrix(test_labels, predicted_classes).numpy()

    # Normalize the confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Convert to percentages
    cm_percent = cm_normalized * 100

    # Plot the normalized confusion matrix
    df_cm = pd.DataFrame(cm_percent, index=range(4), columns=range(4))
    sn.set_theme(font_scale=1.4)  # for label size
    sn.heatmap(df_cm, annot=True, fmt=".2f", annot_kws={"size": 16}, cmap='Blues')  # font size and color map
    plt.title('Normalized Confusion Matrix (Percentages)')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.show()