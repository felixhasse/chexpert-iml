#!/bin/bash
module load devel/python devel/cuda
unset PYTHONPATH
source thesis/bin/activate
cd chexpert-iml
python -m src.train_curriculum_learning
