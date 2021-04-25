import os
import cv2
import sys
import glob
import io
import IPython.display
import logging
import argparse
import numpy as np
import mxnet as mx
import pandas as pd
import PIL.Image
from pathlib import Path
from dotenv import load_dotenv
import torch

from model.emotion import detectemotion as ime
from mxnet_moon.lightened_moon import lightened_moon_feature

# Load path from .env
faceProto ="../model/facenet/opencv_face_detector.pbtxt"
faceModel = "../model/facenet/opencv_face_detector_uint8.pb"
ageProto = "../model/age/age_deploy.prototxt"
ageModel = "../model/age/age_net.caffemodel"
genderProto = "../model/gender/gender_deploy.prototxt"
genderModel = "../model/gender/gender_net.caffemodel"
#pathImg = images
APPROOT = "../"

def load_detection_model():
    # Load face detection model
    faceNet = cv2.dnn.readNet(faceModel, faceProto)
    # Load age detection model
    ageNet = cv2.dnn.readNet(ageModel, ageProto)
    # Load gender detection model
    genderNet = cv2.dnn.readNet(genderModel, genderProto)
    # create instance for emotion detection
    ed = ime.Emotional()




