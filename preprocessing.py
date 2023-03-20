import glob
from typing import List

import numpy as np
from PIL import Image, ImageDraw

from constants import *
from os import path
import os


def convert_jsrt_images():
    """Convert all images from the JRST dataset to png"""
    converted_images_dir = path.join(JSRT_PATH, "png_images")
    image_paths = sorted(glob.glob(path.join(JSRT_PATH, "images", "*.IMG")))
    if not path.exists(converted_images_dir):
        os.makedirs(converted_images_dir)

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
        image.save(fp=path.join(converted_images_dir, image_name + ".png"), format="png")


def generate_jsrt_masks():
    landmark_paths = sorted(glob.glob(path.join(JSRT_PATH, "landmarks", "*.pfs")))

    # Specify mask directories
    mask_base_dir = path.join(JSRT_PATH, "masks")
    left_lung_dir = path.join(mask_base_dir, "left_lung")
    right_lung_dir = path.join(mask_base_dir, "right_lung")
    combined_lung_dir = path.join(mask_base_dir, "both_lungs")
    heart_dir = path.join(mask_base_dir, "heart")

    # Create mask directories if they don't exist
    for directory in [mask_base_dir, left_lung_dir, right_lung_dir, combined_lung_dir, heart_dir]:
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
        right_lung_mask = Image.new(mode="1", size=(1024, 1024))
        left_lung_mask = Image.new(mode="1", size=(1024, 1024))
        combined_lung_mask = Image.new(mode="1", size=(1024, 1024))
        heart_mask = Image.new(mode="1", size=(1024, 1024))

        # Draw masks using landmarks
        draw = ImageDraw.Draw(right_lung_mask)
        draw.polygon(right_lung_landmarks, outline="white", fill="white")

        draw = ImageDraw.Draw(left_lung_mask)
        draw.polygon(left_lung_landmarks, outline="white", fill="white")

        draw = ImageDraw.Draw(heart_mask)
        draw.polygon(heart_landmarks, outline="white", fill="white")

        draw = ImageDraw.Draw(combined_lung_mask)
        draw.polygon(left_lung_landmarks, outline="white", fill="white")
        draw.polygon(right_lung_landmarks, outline="white", fill="white")

        right_lung_mask.save(fp=path.join(right_lung_dir, image_name + ".png"), format="png")
        left_lung_mask.save(fp=path.join(left_lung_dir, image_name + ".png"), format="png")
        combined_lung_mask.save(fp=path.join(combined_lung_dir, image_name + ".png"), format="png")
        heart_mask.save(fp=path.join(heart_dir, image_name + ".png"), format="png")


convert_jsrt_images()
generate_jsrt_masks()
