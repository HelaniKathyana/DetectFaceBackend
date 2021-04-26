import os
import cv2
import sys
import glob
import io
import IPython.display
import logging
import argparse
import numpy as np
import pandas as pd
import torch
import json
import random

import PIL.Image
from pathlib import Path
from dotenv import load_dotenv
import torch
import feature_axis
import pickle

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

from content.interfacegan.models.model_settings import MODEL_POOL
from content.interfacegan.models.pggan_generator import PGGANGenerator
from content.interfacegan.models.stylegan_generator import StyleGANGenerator
from content.interfacegan.utils.manipulator import linear_interpolate

def build_generator(model_name):
    """Builds the generator by model name."""
    gan_type = MODEL_POOL[model_name]['gan_type']
    if gan_type == 'pggan':
        generator = PGGANGenerator(model_name)
    elif gan_type == 'stylegan':
        generator = StyleGANGenerator(model_name)
    return generator

def sample_codes(generator, num, latent_space_type='Z', seed=0):
    """Samples latent codes randomly."""
#     np.random.seed(seed)
    codes = generator.easy_sample(num)
    if generator.gan_type == 'stylegan' and latent_space_type == 'W':
        codes = torch.from_numpy(codes).type(torch.FloatTensor).to(generator.run_device)
        codes = generator.get_value(generator.model.mapping(codes))
    return codes

def imshow(images, col, viz_size=256, name='default'):
    """Shows images in one figure."""
    num, height, width, channels = images.shape
    assert num % col == 0
    row = num // col
    fused_image = np.zeros((viz_size * row, viz_size * col, channels), dtype=np.uint8)
    for idx, image in enumerate(images):
        i, j = divmod(idx, col)
        y = i * viz_size
        x = j * viz_size
        if height != viz_size or width != viz_size:
            image = cv2.resize(image, (viz_size, viz_size))
        fused_image[y:y + viz_size, x:x + viz_size] = image
    fused_image = np.asarray(fused_image, dtype=np.uint8)
    data = io.BytesIO()
    link = 'static/' + name
    print(link)
    PIL.Image.fromarray(fused_image).save(link)
    data.seek(0)
    return name


def load_tl_gan_model():
    """
    Load the linear model (matrix) which maps the feature space
    to the GAN's latent space.
    """
    FEATURE_DIRECTION_FILE = "feature_direction_20181002_044444.pkl"
    with open(FEATURE_DIRECTION_FILE, 'rb') as f:
        feature_direction_name = pickle.load(f)

    # Pick apart the feature_direction_name data structure.
    feature_direction = feature_direction_name['direction']
    feature_names = feature_direction_name['name']
    num_feature = feature_direction.shape[1]
    feature_lock_status = np.zeros(num_feature).astype('bool')

    # Rearrange feature directions using Shaobo's library function.
    feature_direction_disentangled = \
        feature_axis.disentangle_feature_axis_by_idx(
            feature_direction,
            idx_base=np.flatnonzero(feature_lock_status))
    return feature_direction_disentangled, feature_names

def get_random_features(feature_names, seed):
    """
    Return a random dictionary from feature names to feature
    values within the range [40,60] (out of [0,100]).
    """
    #np.random.seed(seed)
    features = dict((name, 40+np.random.randint(0,20)) for name in feature_names)
    features['Male'] = 10
    return features


def selectModel(params):
    model_name = "stylegan_ffhq"  # @param ['pggan_celebahq','stylegan_celebahq', 'stylegan_ffhq']
    latent_space_type = "W"  # @param ['Z', 'W']
    generator = build_generator(model_name)
    # tl_gan_model, feature_names = load_tl_gan_model()
    

    num_samples = 1  # @param {type:"slider", min:1, max:8, step:1}
    noise_seed = 870  # @param {type:"slider", min:0, max:1000, step:1}
    data = []
    latent_space = []
    deleteFiles('static/generate')
    for i in range(0, 4):
        # seed = 2783409
        # features = get_random_features(feature_names, seed)
        # feature_values = np.array([features[name] for name in feature_names])
        # feature_values = (feature_values - 50) / 250
        # Multiply by Shaobo's matrix to get the latent variables.
        # latent_codes = np.dot(tl_gan_model, feature_values)
        # latent_codes = latent_codes.reshape(1, -1)
        latent = sample_codes(generator, num_samples, latent_space_type, noise_seed)
        latent_codes = edit_latent_code(latent, params, model_name, generator, latent_space_type)
        if generator.gan_type == 'stylegan' and latent_space_type == 'W':
            synthesis_kwargs = {'latent_space_type': 'W'}
        else:
            synthesis_kwargs = {}

        images = generator.easy_synthesize(latent_codes, **synthesis_kwargs)['image']
        filename = "generate/generated_" + str(random.randint(0,100)) + ".png"
        data.append(imshow(images, col=num_samples, name=filename))
        latent_space.append(json.dumps(latent.tolist()))
    return [data, latent_space]

def edit_image(latent_codes, params):
    # @title { display-mode: "form", run: "auto" }
    model_name = "stylegan_ffhq"  # @param ['pggan_celebahq','stylegan_celebahq', 'stylegan_ffhq']
    latent_space_type = "W"  # @param ['Z', 'W']
    num_samples = 1  # @param {type:"slider", min:1, max:8, step:1}
    generator = build_generator(model_name)
    new_codes = edit_latent_code(latent_codes, params, model_name, generator, latent_space_type)
    new_images = generator.easy_synthesize(new_codes, **{'latent_space_type': 'W'})['image']
    deleteFiles('static/edit')
    filename = "edit/edited_" + str(random.randint(0,100)) + ".png"
    return imshow(new_images, col=num_samples, name= filename)


def deleteFiles(path):
    dir = path
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))

def mapParams(value):
    if value <= 50:
        return -3 + 3*(value/50)
    else:
        return 3*((value-50)/50)


def edit_latent_code(latent_codes, params, model_name, generator, latent_space_type):
    age = mapParams(params[0])  # @param {type:"slider", min:-3.0, max:3.0, step:0.1}
    eyeglasses = mapParams(params[4]) # @param {type:"slider", min:-2.9, max:3.0, step:0.1}
    gender = mapParams(params[1])  # @param {type:"slider", min:-3.0, max:3.0, step:0.1}
    pose = mapParams(params[2])  # @param {type:"slider", min:-3.0, max:3.0, step:0.1}
    smile = mapParams(params[3]) # @param {type:"slider", min:-3.0, max:3.0, step:0.1}
    print(age)
    ATTRS = ['age', 'eyeglasses', 'gender', 'pose', 'smile']
    boundaries = {}
    for i, attr_name in enumerate(ATTRS):
        boundary_name = f'{model_name}_{attr_name}'
        if generator.gan_type == 'stylegan' and latent_space_type == 'W':
            boundaries[attr_name] = np.load(f'content/interfacegan/boundaries/{boundary_name}_w_boundary.npy')
        else:
            boundaries[attr_name] = np.load(f'content/interfacegan/boundaries/{boundary_name}_boundary.npy')

    new_codes = latent_codes.copy()
    for i, attr_name in enumerate(ATTRS):
        new_codes += boundaries[attr_name] * eval(attr_name)

    return new_codes
