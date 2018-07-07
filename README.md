# Drone_detection-


import os
import cv2
import numpy as np
import tensorflow as tf
import sys


sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the drone detection module we're using
MODEL_NAME = 'inference_graph' 
IMAGE_NAME = 'test_drone.jpg'

# Grab path to current working directory
CWD_PATH = os.getcwd()


# for drone detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# training qovluğu
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

# Path to image
PATH_TO_IMAGE = os.path.join(CWD_PATH,IMAGE_NAME)

# Number of classes the object detector can identify
NUM_CLASSES = 1
