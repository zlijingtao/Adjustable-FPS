#!/bin/bash
cd "$(dirname "$0")"

# FLAGS  --visualize   --presort    --save_computation_two_axis   --use_gpu
# FLAGS **   --presort   --test_dimsort   --test_grid_gcn      --parallel_option
batch_size=24
sort_dim=0
dimsort_range=4
gridgcn_sample_opt=rvs
voxel_size=40


python setup.py --use_gpu --presort --sort_dim=${sort_dim} --batch_size=${batch_size} --dimsort_range=${dimsort_range} --gridgcn_sample_opt=${gridgcn_sample_opt} --voxel_size=${voxel_size}
# python train_semseg.py --model pointnet2_sem_seg --test_area 5 --log_dir pointnet2_sem_seg
python test_semseg.py --log_dir pointnet2_sem_seg --test_area 5