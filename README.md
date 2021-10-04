# MIT-Adobe-FiveK-Pytorch
A Pytorch dataset class implementation for MIT-Adobe FiveK dataset.

Please refer to the dataset website to get the license and other information about the dataset:
https://data.csail.mit.edu/graphics/fivek/

The code consists of:    
`preprocess_5k.py` : Code to convert the ".dng" files into ".jpg".   
`MITAdobe5K.py` : Dataset implementation.   

Note:
1.  Download the dataset here https://data.csail.mit.edu/graphics/fivek/fivek_dataset.tar and extract the dataset. The extracted dataset is assumed in a folder named "MIT-Adobe-5K".
2.  Prepare a folder named "5K-jpgs". This folder should contain subdirectories named "train", "val", and "test". The file `preprocess_5k.py` will split, process, and save the ".jpg" images to these directories. If you want to change directory name or location, please change the variable `DATASET_DIR` and 'TARGET_DIR' in the file.
3.  The file `MITAdobe5K.py` contains the Pytorch class implementation for the dataset. Please adjust the `DATASET_DIR` in accordance to the `preprocess_5k.py` (if you make any change).
