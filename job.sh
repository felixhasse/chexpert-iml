#!/bin/bash
module load devel/python devel/cuda
unset PYTHONPATH
source venv-3.10/bin/activate
cd chexpert-iml/
python train.py
