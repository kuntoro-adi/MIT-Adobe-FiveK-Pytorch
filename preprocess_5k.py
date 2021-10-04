import os
import rawpy
import imageio
import cv2
import numpy as np
import matplotlib.pyplot as plt

DATASET_DIR = '../MIT-Adobe-5K'
TARGET_DIR = '../5K-jpgs'

for root, dirs, files in os.walk(DATASET_DIR):
    for file in files:
        if file.endswith('.dng'):
            path = os.path.join(root, file)
            raw = rawpy.imread(path)
            rgb = raw.postprocess()
            rgb = cv2.resize(rgb, (int(rgb.shape[1] * 0.4), int(rgb.shape[0] * 0.4)))
            img_id = int((file.split('-')[0])[1:])
            split = 'test' if img_id % 5 == 0 else ('val' if img_id % 5 == 1 else 'train')
            imageio.imsave(os.path.join(TARGET_DIR, split, file[0:-4] + '.jpg'), rgb, quality=95)
            print(file, ' --- ', img_id, ':', split)
            
print('Finished.')
