import warnings

warnings.filterwarnings("ignore")

import sys
import numpy as np
import cv2
from time import sleep
from datetime import datetime
from pysolar import solar
import os
import numpy as np
from random import shuffle
from matplotlib import pyplot as plt
import matplotlib as mpl
import pytz
import pandas as pd
import pickle

from pandas import read_csv, read_excel, DataFrame

from skimage.feature import local_binary_pattern as LBP
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

import platform


if platform.uname().system == "Windows":
    q = os.path.join("E:", "usda", "raw", "CookIRCamET")
    p0 = os.path.join(q, "Images", "CookHY2023")
    p1 = os.path.join(q, "Images", "CookHY2024")
    p00 = os.path.join(q, "Images", "CprlHY2023")
    p3 = os.path.join("E:", "usda", "work", "CookIRCamET", "Working")
else:
    q = os.path.join("../../", "raw", "CookIRCamET")
    p0 = os.path.join("../../", "raw", "CookIRCamET", "Images", "CookHY2023")
    p1 = os.path.join("../../", "raw", "CookIRCamET", "Images", "CookHY2024")
    p00 = os.path.join("../../", "raw", "CookIRCamET", "Images", "CprlHY2023")
    p3 = os.path.join("../../", "work", "CookIRCamET", "Working")


n_components = 8

plots = False


def localSD(mat, n):
    mat = np.float32(mat)
    mu = cv2.blur(mat, (n, n))
    mdiff = mu - mat
    mat2 = cv2.blur(np.float64(mdiff * mdiff), (n, n))
    sd = np.float32(cv2.sqrt(mat2))

    return sd


def cornfusion(obs, pred, nclass):
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


def quadrant(img_size, coord):
    if coord[0] > img_size[1] / 2 and coord[1] > img_size[0] / 2:
        return 1
    elif coord[0] > img_size[1] / 2 and coord[1] <= img_size[0] / 2:
        return 4
    elif coord[0] <= img_size[1] / 2 and coord[1] > img_size[0] / 2:
        return 2
    else:
        return 3


def register_ir(ir, v, bgr, warp_mat=None):
    # plt.imshow(v)
    # plt.title('v')
    # plt.colorbar()
    # plt.show()
    # plt.imshow(ir)
    # plt.title('ir')
    # plt.colorbar()
    # plt.show()

    dilate_v = True
    dilate_ir = True
    erode_v = True
    erode_ir = False
    hough_thresh_ir = 36
    if v.shape[1] == 1280:
        hough_thresh_v = 240
    elif v.shape[1] == 1920:
        hough_thresh_v = 360
    if warp_mat is None:
        print("Calculate affine")
        srcXY, dstXY = [], []
        cv2.normalize(ir, ir, 0, 65535, cv2.NORM_MINMAX)
        cv2.normalize(v, v, 0, 255, cv2.NORM_MINMAX)

        ir = np.uint8(np.right_shift(ir, 8, ir))

        # Otsu's thresholding
        xwin = int(v.shape[1] / (1280 / 240)) + 1
        C = -50
        if v.shape[1] == 1920:
            C = -87
        # print(xwin,C)
        # print(v.shape)
        v = cv2.adaptiveThreshold(
            v, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, xwin, C
        )
        # _,v = cv2.threshold(v,245,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        ir = cv2.adaptiveThreshold(
            ir, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 41, 50
        )
        # _,ir = cv2.threshold(ir,137,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        erodewin = int(1920 / (1280 / 2)) + 1

        plt.imshow(v)
        plt.title("v")
        plt.show()
        plt.imshow(ir)
        plt.title("ir")
        plt.show()

        if erode_v:
            kernel = np.ones((erodewin, erodewin), np.uint8)
            v = cv2.erode(v, kernel, iterations=1)

        if erode_ir:
            kernel = np.ones((erodewin, erodewin), np.uint8)
            ir = cv2.erode(ir, kernel, iterations=1)

        if dilate_v:
            kernel = np.ones((erodewin, erodewin), np.uint8)
            v = cv2.dilate(v, kernel, iterations=1)

        if dilate_ir:
            kernel = np.ones((erodewin, erodewin), np.uint8)
            ir = cv2.dilate(ir, kernel, iterations=1)

        plt.imshow(v)
        plt.title("v")
        plt.show()
        plt.imshow(ir)
        plt.title("ir")
        plt.show()

        line_params_v = {"slope": [], "intercept": []}
        lines_v = cv2.HoughLines(v, 1, np.pi / 180, hough_thresh_v)
        slope = np.zeros(lines_v.shape[0])
        intercept = np.zeros(lines_v.shape[0])
        four_v = np.zeros((lines_v.shape[0], 4))
        ii = 0
        for line in lines_v:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + v.shape[1] * (-b))
            y1 = int(y0 + v.shape[1] * (a))
            x2 = int(x0 - v.shape[1] * (-b))
            y2 = int(y0 - v.shape[1] * (a))
            cv2.line(bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
            four_v[ii, :] = np.array([x1, y1, x2, y2])
            if x2 == x1:
                slope[ii] = (y2 - y1) / (x2 - x1 + 0.0000001 * np.random.randn(1))
            else:
                slope[ii] = (y2 - y1) / (x2 - x1)
            intercept[ii] = y2 - slope[ii] * x2
            ii = ii + 1

        line_params_v["slope"] = slope
        line_params_v["intercept"] = intercept

        lines_ir = cv2.HoughLines(ir, 1, np.pi / 180, hough_thresh_ir)
        line_params_ir = {"slope": [], "intercept": []}
        slope = np.zeros(lines_ir.shape[0])
        intercept = np.zeros(lines_ir.shape[0])
        four_ir = np.zeros((lines_ir.shape[0], 4))
        ii = 0
        for line in lines_ir:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 200 * (-b))
            y1 = int(y0 + 200 * (a))
            x2 = int(x0 - 200 * (-b))
            y2 = int(y0 - 200 * (a))
            cv2.line(bgr, (x1, y1), (x2, y2), (255, 0, 0), 2)
            four_ir[ii, :] = np.array([x1, y1, x2, y2])
            if x2 == x1:
                slope[ii] = (y2 - y1) / (x2 - x1 + 0.0000001 * np.random.randn(1))
            else:
                slope[ii] = (y2 - y1) / (x2 - x1)
            intercept[ii] = y2 - slope[ii] * x2
            ii = ii + 1
        line_params_ir["slope"] = slope
        line_params_ir["intercept"] = intercept

        n = line_params_v["slope"].shape[0]
        combos = int((n * (n - 1)) / 2)
        corner_v = np.zeros((combos, 2))
        kk = 0
        for ii in range(line_params_v["slope"].shape[0]):
            for jj in range(ii, line_params_v["slope"].shape[0]):
                if ii != jj:
                    if (
                        np.abs(
                            np.abs(line_params_v["slope"][ii])
                            * np.abs(line_params_v["slope"][jj])
                            - 1
                        )
                        < 0.1
                    ):
                        x = (
                            line_params_v["intercept"][jj]
                            - line_params_v["intercept"][ii]
                        ) / (line_params_v["slope"][ii] - line_params_v["slope"][jj])
                        y = (
                            line_params_v["slope"][ii] * x
                            + line_params_v["intercept"][ii]
                        )
                        corner_v[kk, 0] = x
                        corner_v[kk, 1] = y
                        kk = kk + 1
                        try:
                            cv2.circle(
                                bgr,
                                (int(x), int(y)),
                                radius=3,
                                color=(0, 0, 0),
                                thickness=-1,
                            )
                        except:
                            pass
                    else:
                        x = np.nan
                        y = np.nan
                        corner_v[kk, 0] = x
                        corner_v[kk, 1] = y
                        kk = kk + 1
        n = line_params_ir["slope"].shape[0]
        combos = int((n * (n - 1)) / 2)
        corner_ir = np.zeros((combos, 2))
        kk = 0
        for ii in range(line_params_ir["slope"].shape[0]):
            for jj in range(ii, line_params_ir["slope"].shape[0]):
                if ii != jj:
                    if (
                        np.abs(
                            np.abs(line_params_ir["slope"][ii])
                            * np.abs(line_params_ir["slope"][jj])
                            - 1
                        )
                        < 0.1
                    ):
                        x = (
                            line_params_ir["intercept"][jj]
                            - line_params_ir["intercept"][ii]
                        ) / (line_params_ir["slope"][ii] - line_params_ir["slope"][jj])
                        y = (
                            line_params_ir["slope"][ii] * x
                            + line_params_ir["intercept"][ii]
                        )
                        corner_ir[kk, 0] = x
                        corner_ir[kk, 1] = y
                        kk = kk + 1
                        try:
                            cv2.circle(
                                bgr,
                                (int(x), int(y)),
                                radius=3,
                                color=(255, 255, 255),
                                thickness=-1,
                            )
                        except:
                            pass
                    else:
                        x = np.nan
                        y = np.nan
                        corner_ir[kk, 0] = x
                        corner_ir[kk, 1] = y
                        kk = kk + 1

        corner_ir = corner_ir[np.argwhere(np.isnan(corner_ir[:, 0]) == False)[:, 0],]
        corner_v = corner_v[np.argwhere(np.isnan(corner_v[:, 0]) == False)[:, 0],]

        plt.imshow(bgr)
        plt.show()
        plt.imshow(ir)
        plt.show()

        q_v = np.zeros(corner_v.shape[0])
        q_ir = np.zeros(corner_ir.shape[0])
        for ii in range(corner_v.shape[0]):
            q_v[ii] = quadrant(bgr.shape[0:2], corner_v[ii, :])
        for jj in range(corner_ir.shape[0]):
            q_ir[jj] = quadrant(ir.shape[0:2], corner_ir[jj, :])

        q_list = list()
        for ii in range(corner_v.shape[0]):
            for jj in range(corner_ir.shape[0]):
                if q_v[ii] == q_ir[jj]:
                    srcXY.append(corner_ir[jj, :])
                    dstXY.append(corner_v[ii, :])
                    q_list.append(q_v[ii])

        srcXY = np.float32(np.array(srcXY))
        dstXY = np.float32(np.array(dstXY))
        q_list = np.float32(np.array(q_list))

        src = np.zeros((4, 2))
        dst = np.zeros((4, 2))

        for i in range(4):
            src[i, :] = np.mean(srcXY[q_list == i + 1, :], axis=0)
            dst[i, :] = np.mean(dstXY[q_list == i + 1, :], axis=0)
        warp_mat = cv2.estimateAffine2D(src, dst)
        warp_dst = None
    else:
        # print('Apply affine')
        warp_dst = cv2.warpAffine(
            ir, warp_mat[0], (v.shape[1], v.shape[0]), flags=cv2.INTER_NEAREST
        )
    return warp_mat, warp_dst


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
