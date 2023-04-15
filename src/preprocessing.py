import glob
import shutil
from typing import List

import numpy as np
from PIL import Image, ImageDraw, ImageChops

from materials.constants import *
from os import path
import os


def convert_jsrt_images():
    """Convert all images from the JRST dataset to png"""
    heart_images_dir = path.join(SEGMENTATION_DATASET_PATH, "heart", "images")
    lung_images_dir = path.join(SEGMENTATION_DATASET_PATH, "lung", "images")

    image_paths = sorted(glob.glob(path.join(JSRT_PATH, "images", "*.IMG")))
    for directory in [heart_images_dir, lung_images_dir]:
        if not path.exists(directory):
            os.makedirs(directory)

    for image_path in image_paths:
        image_name = image_path.split("/")[-1].split(".")[0]
        print(f"Converting image {image_name}")

        # Adapted from https://github.com/harishanand95/jsrt-parser/blob/master/jsrt.py
        # Image is of size 2048x2048 in gray scale stored in 16 bit unsigned int in big endian format.
        raw_image = np.fromfile(image_path, dtype=">i2").reshape((2048, 2048))

        # Normalize and invert for 8-bit color
        raw_image = 255 - (raw_image * (255 / 4095))

        image = Image.fromarray(raw_image).convert("L")

        # Resize to 1024x1024, which corresponds to the mask dimensions
        image = image.resize(size=(1024, 1024))
        image.save(fp=path.join(heart_images_dir, image_name + ".png"), format="png")
        image.save(fp=path.join(lung_images_dir, image_name + ".png"), format="png")


def generate_jsrt_masks():
    landmark_paths = sorted(glob.glob(path.join(JSRT_PATH, "landmarks", "*.pfs")))

    # Specify mask directories
    lung_dir = path.join(SEGMENTATION_DATASET_PATH, "lung", "masks")
    heart_dir = path.join(SEGMENTATION_DATASET_PATH, "heart", "masks")

    # Create mask directories if they don't exist
    for directory in [lung_dir, heart_dir]:
        if not path.exists(directory):
            os.makedirs(directory)

    for landmark_path in landmark_paths:
        right_lung_landmarks: List[tuple] = []
        left_lung_landmarks: List[tuple] = []
        heart_landmarks: List[tuple] = []

        with open(landmark_path, "r") as file:
            image_name = landmark_path.split("/")[-1].split(".")[0]
            print(f"Generating masks for image {image_name}")
            lines = file.readlines()

            lines = [line.replace(" ", "") for line in lines]
            lines = [line.replace("\n", "") for line in lines]

            right_lung_landmarks_start = lines.index("[Label=rightlung]")
            left_lung_landmarks_start = lines.index("[Label=leftlung]")
            heart_landmarks_start = lines.index("[Label=heart]")
            # We don't use the clavicle masks, just used as the limit for the search for heart landmarks
            right_clavicle_start = lines.index("[Label=rightclavicle]")

            for i in range(right_lung_landmarks_start + 2, left_lung_landmarks_start):

                # Make sure that only lines with landmarks are included
                if "{" in lines[i] and "}" in lines[i]:
                    # Extract (x,y) tuple from the line
                    str_coordinates = lines[i].split("{")[1].split("}")[0].split(",")
                    coordinates = tuple(float(value) for value in str_coordinates)
                    right_lung_landmarks.append(coordinates)

            for i in range(left_lung_landmarks_start + 2, heart_landmarks_start):

                # Make sure that only lines with landmarks are included
                if "{" in lines[i] and "}" in lines[i]:
                    # Extract (x,y) tuple from the line
                    str_coordinates = lines[i].split("{")[1].split("}")[0].split(",")
                    coordinates = tuple(float(value) for value in str_coordinates)
                    left_lung_landmarks.append(coordinates)

            for i in range(heart_landmarks_start + 2, right_clavicle_start):

                # Make sure that only lines with landmarks are included
                if "{" in lines[i] and "}" in lines[i]:
                    # Extract (x,y) tuple from the line
                    str_coordinates = lines[i].split("{")[1].split("}")[0].split(",")
                    coordinates = tuple((float(value) for value in str_coordinates))
                    heart_landmarks.append(coordinates)

        # Initialize masks as blank (all black) 1-bit color images
        combined_lung_mask = Image.new(mode="1", size=(1024, 1024))
        heart_mask = Image.new(mode="1", size=(1024, 1024))

        # Draw masks using landmarks
        draw = ImageDraw.Draw(heart_mask)
        draw.polygon(heart_landmarks, outline="white", fill="white")

        draw = ImageDraw.Draw(combined_lung_mask)
        draw.polygon(left_lung_landmarks, outline="white", fill="white")
        draw.polygon(right_lung_landmarks, outline="white", fill="white")

        combined_lung_mask.save(fp=path.join(lung_dir, image_name + ".png"), format="png")
        heart_mask.save(fp=path.join(heart_dir, image_name + ".png"), format="png")


def process_montgomery_dataset():
    image_paths = sorted(glob.glob(path.join(MONTGOMERY_PATH, "CXR_png", "*.png")))
    left_lung_paths = sorted(glob.glob(path.join(MONTGOMERY_PATH, "ManualMask", "leftMask", "*.png")))
    right_lung_paths = sorted(glob.glob(path.join(MONTGOMERY_PATH, "ManualMask", "rightMask", "*.png")))

    lung_images_dir = path.join(SEGMENTATION_DATASET_PATH, "lung", "images")
    lung_mask_dir = path.join(SEGMENTATION_DATASET_PATH, "lung", "masks")

    for image_path in image_paths:
        image_name = os.path.basename(image_path)
        dest_path = os.path.join(lung_images_dir, image_name)
        shutil.copyfile(image_path, dest_path)

    for i in range(len(left_lung_paths)):
        image_name = os.path.basename(left_lung_paths[i])
        left_lung_mask = Image.open(left_lung_paths[i]).convert("1")
        right_lung_mask = Image.open(right_lung_paths[i]).convert("1")
        combined_mask = ImageChops.logical_or(left_lung_mask, right_lung_mask)
        combined_mask.save(os.path.join(SEGMENTATION_DATASET_PATH, "lung", "masks", image_name))


convert_jsrt_images()
generate_jsrt_masks()
process_montgomery_dataset()
print("Done")
