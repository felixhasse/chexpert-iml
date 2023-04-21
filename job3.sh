#!/bin/bash
module load devel/python devel/cuda
unset PYTHONPATH
source thesis/bin/activate
cd chexpert-iml
python -m src.segment_chexpert -n DeepLabV3_combined_aug_2
