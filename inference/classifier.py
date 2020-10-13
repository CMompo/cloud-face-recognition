#!/usr/bin/env python

import os
import joblib
import argparse
from PIL import Image
from util import draw_bb_on_img
from constants import MODEL_PATH
from face_recognition import preprocessing


def parse_args():
    parser = argparse.ArgumentParser(prog='Face Classifier',
    description='Script for detecting and classifying faces on user-provided image. This script will process image, draw '
        'bounding boxes and labels on image and display it. It will also optionally save that image.')
    parser.add_argument('--image-path', required=True, help='Path to image file.')
    parser.add_argument('--save-dir', help='If save dir is provided image will be saved to specified directory.')
    parser.add_argument('--min-conf', help='Only use face predictions that have a confidence of at least the specified value (0-1).')
    parser.add_argument('--fast', action='store_true', help='Enable Low-Res fast mode.')
    return parser.parse_args()


def recognise_faces(img, args):
    faces = joblib.load(MODEL_PATH)(img)
    if args.min_conf:
        faces = [face for face in faces if face.top_prediction.confidence > float(args.min_conf)]
    if faces:
        draw_bb_on_img(faces, img)
    return faces, img


def main():
    args = parse_args()
    preprocess = preprocessing.ExifOrientationNormalize()
    img = Image.open(args.image_path)
    filename = img.filename
    if args.fast:
        width, height = img.size
        factor = 512/width
        size = [round(width*factor), round(height*factor)]
        img = img.resize(size, Image.BILINEAR)
    img = preprocess(img)
    img = img.convert('RGB')

    faces, img = recognise_faces(img, args)
    if not faces:
        print('No faces found in this image.')

    if args.save_dir:
        basename = os.path.basename(filename)
        name = basename.split('.')[0]
        ext = basename.split('.')[1]
        img.save('{}_tagged.{}'.format(name, ext))

    img.show()


if __name__ == '__main__':
    main()
