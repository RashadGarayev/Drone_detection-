# Drone_detection-
#Qeyd
kitabxananı istifadə etməmişdən öncə pipeline.config faylını notepad++ ilə açın və aşağıdakı sətirlərdə dəyişiklik edin
1. 109 cu sətrdə qeyd olunan fine_tune_checkpoint: "C:/models/research/object_detection/faster_rcnn_inception_v2_coco_2018_01_28/model.ckpt"
qovluğ yolunu tensorflo yüklədiyiniz ünvanla eyniləşdirin.yuxarıdakı sətirdə C:/models/research/object_detection -şəxsi kompyuterimdə C-yə yüklədiyim üçün nümunə olaraq göstərmişəm.

2. 114 cü sətirdə də həmçinin qovluğ yolunu təyin olunan yerə uyğun dəyişin
label_map_path: "C:/models/research/object_detection/training/labelmap.pbtxt"

3.Dəyişiklik edəcəyiniz digər sətrlər

116-cı sətr   input_path: "C:/models/research/object_detection/train.record"

125 ci sətr   label_map_path: "C:/models/research/object_detection/training/labelmap.pbtxt"

və son olaraq 

129cu sətr

tf_record_input_reader {
    input_path: "C:/models/research/object_detection/test.record"
  }






Aşağıda göstərilən .py faylında kitabxananı istifadə üçün 46-cı sətrdə inference_graph olaraq yazılıq.(dronları təyin etmək üçün kitabxanadır)
47-ci sətrdə göstərilən test_drone.jpg faylı isə sizin əlavə edəcəyiniz rəsmdir-nümunə üçün göstərilib
58-ci sətrdə qeyd olunan labelmap.pbtxt faylı training qovluğundadır.training qovluğunu object_detection qovluğuna çıxarın-kitabxana daxilində olmamalıdır
65-ci sətrdə qeyd olunan NUM_Classes 1 olaraq qalacaq,çünki təyin edəcəyimiz obyekt sadəcə drondur

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

# training 
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

# Path to image
PATH_TO_IMAGE = os.path.join(CWD_PATH,IMAGE_NAME)

# Number of classes the object detector can identify
NUM_CLASSES = 1
