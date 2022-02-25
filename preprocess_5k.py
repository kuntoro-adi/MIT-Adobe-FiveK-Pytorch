import os
import rawpy
import imageio
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

# Where the dataset root located
DATASET_DIR = '../Datasets/fivek_dataset'
# Target location to save image samples
TARGET_DIR = '../Datasets/5k-jpgs'

# Determine split (train val test)
random.seed(7)
split = list(range(1,5001))
random.shuffle(split)
train_split, validation_split, test_split = split[0:4000], split[4000:4500], split[4500:5000]

print('train_split[:6]', train_split[:6])
print('validation_split[:6]', validation_split[:6])
print('test_split[:6]', test_split[:6])
print()

# Log file
log_file = "file_name, width, height, width_n, height_n\n"
counter = 0

# Iterate samples
for root, dirs, files in os.walk(DATASET_DIR):
    for file_name in files:
        # Process dng image only
        if file_name.endswith('.dng'):
            # Load and process
            path = os.path.join(root, file_name)
            raw = rawpy.imread(path)
            rgb = raw.postprocess()
            # Adjust size
            width, height = int(rgb.shape[1]), int(rgb.shape[0]) 
            width_n, height_n = int(width / 128) * 32 , int(height / 128) * 32 # quarter the original size, divisible by 32
            rgb = cv2.resize(rgb, (width_n, height_n))
            # Determine split (training, validation, or testing)
            img_id = int((file_name.split('-')[0])[1:])
            split = 'test' if img_id in test_split else ('val' if img_id in validation_split else 'train')
            # Save image
            imageio.imsave(os.path.join(TARGET_DIR, split, file_name[0:-4] + '.jpg'), rgb, quality=95)
            print(counter, '>', file_name, ' Id:', img_id, ' Split:', split)
            log_file += "{},{},{},{},{}\n".format(file_name, str(width), str(height), str(width_n), str(height_n))
            counter += 1

            #if counter == 20:
            #    break

    #if counter == 20:
    #    break

# save logfile
with open("log_file.csv", "w") as fhandle:
    fhandle.write(log_file)

print('Finished.')
