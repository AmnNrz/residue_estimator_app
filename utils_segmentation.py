import numpy as np
import cv2

from skimage.feature import local_binary_pattern as LBP

def localSD(mat, n):
    mat = np.float32(mat)
    mu = cv2.blur(mat, (n, n))
    mdiff = mu - mat
    mat2 = cv2.blur(np.float64(mdiff * mdiff), (n, n))
    sd = np.float32(cv2.sqrt(mat2))

    return sd

def confusion(obs, pred, nclass):
    M = np.zeros((nclass, nclass))
    for i in range(obs.shape[0]):
        o = obs[i]
        p = pred[i]
        M[o, p] = M[o, p] + 1
    correct = sum(obs == pred)
    total = len(pred)
    M = M / np.sum(np.sum(M))
    recall = np.diag(M) / np.sum(M, axis=1)
    precis = np.diag(M) / np.sum(M, axis=0)

    f1 = recall * precis / (recall + precis) * 2
    f1_weighted = np.sum(f1 * np.sum(M, axis=1))

    return M, f1_weighted, correct / total

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
