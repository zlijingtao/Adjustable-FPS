#!/bin/bash
cd "$(dirname "$0")"

cd data_utils
python collect_indoor3d_data.py

cd ..

python setup.py --use_gpu
python train_semseg.py --model pointnet2_sem_seg --test_area 5 --log_dir pointnet2_sem_seg
python test_semseg.py --log_dir pointnet2_sem_seg --test_area 5 --visual