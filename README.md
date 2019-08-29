# SAR ship detection
This repository contains YOLOv3 software for SAR ship detection.
# Description
The repository contains training, testing, and inference code in PyTorch. 
# Requirements
* Python 3.6 or later version
* Torch
* Numpy
* OpenCV-Python
* tqdm
# Required text files
* ship-obj.data contains path to training and validation data path, number of class, and path to ship-ob.names (class name).
* train.txt : path to the training data.
* validation.txt : path to validation data.
# Labels
* Bounding box labels are created from X and Y coordinates of objects center and width and hight coordinates of objects in txt format.
* Each label txt file has five columns including class index, X, Y, W, H.
* Labels and image files are placed in the same folder.
# CFG file
The cfg file contains the structure of the network. It should be updated based on the number of classes.
# Weights
The best and last weights are saved in weights folder.
# Training
Start training: python3 train.py to begin training on SAR data. Each epoch trains on SAR images and test on SAR validation set.
# Inference
Run detect.py on inference images. Put your inference images in a file named samples in the data folder.
# Evaluation
Run python3 test.py --weights weights/best.pt to test best checkpoint.
Run python3 checkpoint.py to get the best fitness value and epochs of the best checkpoint.
