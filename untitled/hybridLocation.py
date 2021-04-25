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


def selectModel():
    model_name = "stylegan_ffhq"  # @param ['pggan_celebahq','stylegan_celebahq', 'stylegan_ffhq']
    latent_space_type = "W"  # @param ['Z', 'W']
    generator = build_generator(model_name)

    ATTRS = ['age', 'eyeglasses', 'gender', 'pose', 'smile']
    boundaries = {}
    for i, attr_name in enumerate(ATTRS):
        boundary_name = f'{model_name}_{attr_name}'
        if generator.gan_type == 'stylegan' and latent_space_type == 'W':
            boundaries[attr_name] = np.load(f'content/interfacegan/boundaries/{boundary_name}_w_boundary.npy')
        else:
            boundaries[attr_name] = np.load(f'content/interfacegan/boundaries/{boundary_name}_boundary.npy')

    num_samples = 1  # @param {type:"slider", min:1, max:8, step:1}
    noise_seed = 870  # @param {type:"slider", min:0, max:1000, step:1}
    data = []
    latent_space = []
    for i in range(0, 4):
        latent_codes = sample_codes(generator, num_samples, latent_space_type, noise_seed)
        if generator.gan_type == 'stylegan' and latent_space_type == 'W':
            synthesis_kwargs = {'latent_space_type': 'W'}
        else:
            synthesis_kwargs = {}

        images = generator.easy_synthesize(latent_codes, **synthesis_kwargs)['image']
        data.append(imshow(images, col=num_samples, name='face' + str(i) + '.png'))
        latent_space.append(json.dumps(latent_codes.tolist()))
    return [data, latent_space]

def edit_image(latent_codes):
    # @title { display-mode: "form", run: "auto" }

    age = 1  # @param {type:"slider", min:-3.0, max:3.0, step:0.1}
    eyeglasses = 1  # @param {type:"slider", min:-2.9, max:3.0, step:0.1}
    gender = 0  # @param {type:"slider", min:-3.0, max:3.0, step:0.1}
    pose = 0  # @param {type:"slider", min:-3.0, max:3.0, step:0.1}
    smile = -2 # @param {type:"slider", min:-3.0, max:3.0, step:0.1}

    model_name = "stylegan_ffhq"  # @param ['pggan_celebahq','stylegan_celebahq', 'stylegan_ffhq']
    latent_space_type = "W"  # @param ['Z', 'W']
    generator = build_generator(model_name)

    ATTRS = ['age', 'eyeglasses', 'gender', 'pose', 'smile']
    boundaries = {}
    for i, attr_name in enumerate(ATTRS):
        boundary_name = f'{model_name}_{attr_name}'
        if generator.gan_type == 'stylegan' and latent_space_type == 'W':
            boundaries[attr_name] = np.load(f'content/interfacegan/boundaries/{boundary_name}_w_boundary.npy')
        else:
            boundaries[attr_name] = np.load(f'content/interfacegan/boundaries/{boundary_name}_boundary.npy')

    num_samples = 1  # @param {type:"slider", min:1, max:8, step:1}

    new_codes = latent_codes.copy()
    for i, attr_name in enumerate(ATTRS):
        new_codes += boundaries[attr_name] * eval(attr_name)

    new_images = generator.easy_synthesize(new_codes, **{'latent_space_type': 'W'})['image']
    imshow(new_images, col=num_samples, name="edited.png")
