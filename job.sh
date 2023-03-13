#!/bin/bash
module load devel/python devel/cuda
unset PYTHONPATH
source thesis/bin/activate
cd chexpert-iml/
python train_lung_segmentation.py
