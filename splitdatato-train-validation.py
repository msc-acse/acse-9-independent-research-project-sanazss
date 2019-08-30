import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import gdal
import glob
import torch
import re
import tifffile as tiff
from os import listdir
from os.path import isfile, join
import pandas as pd
from sklearn.model_selection import train_test_split


train_image_dir_1 = "/redresearch/ssalati/Sanaz/label_tiles_smallsize" #path to labels

train_image_dir = "/redresearch/ssalati/Sanaz/finaldataset" #path to images

images = [(train_image_dir +f) for f in listdir(train_image_dir) if isfile(join(train_image_dir,f))] #read images

boxes = [(train_image_dir_1 +f) for f in listdir(train_image_dir_1) if isfile(join(train_image_dir_1,f))] #read labels

a = pd.DataFrame(np.column_stack([images, boxes]), columns= ['images' , 'boxes']) #create dataframe

b = a.reindex(np.random.permutation(a.index)) #shuffle the matrix

a1 = b.sort_values(by = 'images')['images'].reset_index() #sorting
a2 = b.sort_values(by='boxes')['boxes'].reset_index() #sorting

b['images'] = a1['images']
b['boxes'] = a2['boxes']
del a1, a2

a_train, a_test = train_test_split(b, test_size = 0.2, random_state = 42) #splitting dataset to train and validation

test = a_test.filter(regex = 'images') #filter test images
test.to_csv('/redresearch/ssalati/Sanaz/finalsplit/test.txt', index = False, header = None) #save the path of test images to csv file
test_label = a_test.filter(regex = 'boxes') #filter labels
test_label.to_csv('/redresearch/ssalati/Sanaz/finalsplit/test_label.txt', index = False, header = None) #save path of test labels


train = a_train.filter(regex = 'images') #filter train images
train.to_csv('/redresearch/ssalati/Sanaz/finalsplit/train.txt', index = False, header = None) #save path of labels to csv file
train_label = a_train.filter(regex = 'boxes') #filter train labels
train_label.to_csv('/redresearch/ssalati/Sanaz/finalsplit/train_label.txt', index = False, header = None) #save path of train label to csv file

#filtering files based on created csv files for spliting data
import os, shutil,sys
from tkinter import filedialog
from tkinter import *
from pathlib import Path
import glob

root = Tk()
root.withdraw()

filePath = filedialog.askopenfilename()
folderPath = filedialog.askdirectory()
destination = filedialog.askdirectory()

filesofinterest = []

with open(filePath, "r") as fh: #csv files of path
    for row in fh:
        a = row.split("/")[-1]
        b = a.strip("label_tiles_smallsize") #strip the file name
        filesofinterest.append(b.strip())
       
print(filesofinterest)        
for filename in os.listdir(folderPath): #find files based on csv files
    if filename in filesofinterest:
        filename = os.path.join(folderPath, filename )
        print(filename)
        shutil.copy( filename, destination) #copy files to new destination
    else:
        print("filename doesnt exist")

#filtering image files based on created labels id        
tilePath = '/redresearch/ssalati/Sanaz/tiles_smallsize'
filepath = '/redresearch/ssalati/Sanaz/label_tiles_smallsize'
tiles = glob.glob(os.path.join(tilePath, "*.tif"))
labels = glob.glob(os.path.join(filepath, "*.txt"))

for label in labels:
    
    name = label.split('/')[-1]
    name = name[:-4]
    for tile in tiles:
        
        if name + ".tif" == tile.split('/')[-1]:
            shutil.copy(tile, '/redresearch/ssalati/Sanaz/finaldataset')
            
 #drawing bounding boxes of ground truth on images
import glob
import cv2
import os
imagepath ='/redresearch/ssalati/Sanaz/finaldataset/037989.000361.tif'
labelpath = '/redresearch/ssalati/Sanaz/label_tiles_smallsize/037989.000361.txt'


img = cv2.imread(imagepath, cv2.IMREAD_ANYDEPTH|cv2.IMREAD_UNCHANGED) #read the image
img = np.stack((img,)*3, axis =2) #stack the channel
fig, ax = plt.subplots(1, figsize=(12,12))
ax.imshow(img, cmap = 'gray') 

x, y, w, h = [], [], [], []

with open(labelpath, "r") as fh:
    for rows in fh:
        coord = rows.split()
        x.append(float(coord [1])) #x coordinates
        y.append(float(coord [2])) #y coordinates
        w.append(float(coord [3])) #width coordinates
        h.append(float(coord [4])) #height coordinates
        for box in list(zip(x,y,w,h)): #creating bounding boxes from coordinates
            box = patches.Rectangle(((box[0]-(box[2]/2))*256, (box[1]-(box[3]/2))*256), box[2]*256, box[3]*256, linewidth=1, edgecolor ='r', facecolor = 'none')
            ax.add_patch(box)
plt.show()
   


#counting number of ships in training and validation data
lines = 0
with open("/redresearch/ssalati/Document.txt", 'r') as f:
    for line in f:
        lines += 1
print(lines)

