#!/bin/bash
cd "$(dirname "$0")"

python setup.py --use_gpu
python train_cls.py --model pointnet2_cls_msg --normal --log_dir pointnet2_cls_msg
python test_cls.py --normal --log_dir pointnet2_cls_msg