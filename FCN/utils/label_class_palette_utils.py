import tensorflow as tf
import numpy as np
import os
import sys
path = os.path.abspath(os.path.join('..'))
sys.path.append(path)

#the color correspond to the label
classes = {'aeroplane': 1,  'bicycle': 2,  'bird': 3,  'boat': 4,
           'bottle': 5,  'bus': 6,  'car': 7,  'cat': 8,
           'chair': 9,  'cow': 10, 'diningtable': 11, 'dog': 12,
           'horse': 13, 'motorbike': 14, 'person': 15, 'potted-plant': 16,
           'sheep': 17, 'sofa': 18, 'train': 19, 'tv/monitor': 20}


classes_reverse = {v: k for k, v in classes.items()}


palette_label = [[0,   0,   0],
           [128,   0,   0],
           [0, 128,   0],
           [128, 128,   0],
           [0,   0, 128],
           [128,   0, 128],
           [0, 128, 128],
           [128, 128, 128],
           [64,   0,   0],
           [192,   0,   0],
           [64, 128,   0],
           [192, 128,   0],
           [64,   0, 128],
           [192,   0, 128],
           [64, 128, 128],
           [192, 128, 128],
           [0,  64,   0],
           [128,  64,   0],
           [0, 192,   0],
           [128, 192,   0],
           [0,  64, 128]]


