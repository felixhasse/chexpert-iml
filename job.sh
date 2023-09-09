#!/bin/bash
module load devel/python devel/cuda compiler/intel
unset PYTHONPATH
source thesis/bin/activate
cd chexpert-iml
python -m src.train_baseline -p 10%_train_data
