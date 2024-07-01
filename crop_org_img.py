# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.1
#   kernelspec:
#     display_name: res_app
#     language: python
#     name: python3
# ---

import cv2 
import numpy as np
import os

# +
path_to_raw = ("/mnt/C250892050891BF3/BioAg_Tillage_data/Residue_%_from "
                   "image/BioAgUavResidue/Amin/")

path_to_cropped = (
    "/mnt/C250892050891BF3/BioAg_Tillage_data/Residue_%_from image/"
    "cropped_org_images/"
)

raw_folders = [
    "Kincaid-conventional_1m_20220420",
    "Ritzville2-SprWheat1m20220329",
    "Ritzville3-WheatFallow1pass1m20220329",
]

for folder in raw_folders: 
    path_to_imgs = path_to_raw + folder + "/"
    folder_imgs = os.listdir(path_to_imgs)
    img_list = [img for img in folder_imgs if img.lower().endswith(".jpg")]
    print(img_list)
    for img in img_list: 
        image = cv2.imread(path_to_imgs + img, cv2.IMREAD_UNCHANGED)
        start_pixel, end_pixel = 0, 2048
        image = image[start_pixel:end_pixel, start_pixel:end_pixel]
        os.makedirs(path_to_cropped + folder, exist_ok=True)
        cv2.imwrite(path_to_cropped + folder + "/" + img + ".jpg", image)

