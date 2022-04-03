#!/bin/bash
cd "$(dirname "$0")"

# FLAGS  --visualize   --presort    --save_computation_two_axis   --use_gpu
# FLAGS **   --presort   --test_dimsort   --test_grid_gcn      --parallel_option
batch_size=24
sort_dim=0
dimsort_range=4
gridgcn_sample_opt=rvs
voxel_size=40

# #Run baseline
# python setup.py --use_gpu --sort_dim=${sort_dim} --batch_size=${batch_size} --dimsort_range=${dimsort_range} --gridgcn_sample_opt=${gridgcn_sample_opt} --voxel_size=${voxel_size}
# python test_semseg.py --log_dir pointnet2_sem_seg --test_area 5

# #Run parallel baseline
# python setup.py --use_gpu --parallel_option --presort --sort_dim=${sort_dim} --batch_size=${batch_size} --dimsort_range=${dimsort_range} --gridgcn_sample_opt=${gridgcn_sample_opt} --voxel_size=${voxel_size}
# python test_semseg.py --log_dir pointnet2_sem_seg --test_area 5

# #Run parallel baseline (no-presort)
# python setup.py --use_gpu --parallel_option --sort_dim=${sort_dim} --batch_size=${batch_size} --dimsort_range=${dimsort_range} --gridgcn_sample_opt=${gridgcn_sample_opt} --voxel_size=${voxel_size}
# python test_semseg.py --log_dir pointnet2_sem_seg --test_area 5

# #Run grid-gcn rvs
# gridgcn_sample_opt=rvs
# voxel_size=40
# python setup.py --use_gpu --test_grid_gcn --parallel_option --presort --sort_dim=${sort_dim} --batch_size=${batch_size} --dimsort_range=${dimsort_range} --gridgcn_sample_opt=${gridgcn_sample_opt} --voxel_size=${voxel_size}
# python test_semseg.py --log_dir pointnet2_sem_seg --test_area 5

# #Run grid-gcn cas
# gridgcn_sample_opt=cas
# voxel_size=40
# python setup.py --use_gpu --test_grid_gcn --parallel_option --presort --sort_dim=${sort_dim} --batch_size=${batch_size} --dimsort_range=${dimsort_range} --gridgcn_sample_opt=${gridgcn_sample_opt} --voxel_size=${voxel_size}
# python test_semseg.py --log_dir pointnet2_sem_seg --test_area 5

#Run RPS
python setup.py --use_gpu --test_rps --sort_dim=${sort_dim} --batch_size=${batch_size} --dimsort_range=${dimsort_range} --gridgcn_sample_opt=${gridgcn_sample_opt} --voxel_size=${voxel_size}
python test_semseg.py --log_dir pointnet2_sem_seg --test_area 5

# #Run sequential dimsort
# dimsort_range=4
# python setup.py --use_gpu --test_dimsort --presort --sort_dim=${sort_dim} --batch_size=${batch_size} --dimsort_range=${dimsort_range} --gridgcn_sample_opt=${gridgcn_sample_opt} --voxel_size=${voxel_size}
# python test_semseg.py --log_dir pointnet2_sem_seg --test_area 5

#Run sequential dimsort
dimsort_range=8
python setup.py --use_gpu --test_dimsort --presort --sort_dim=${sort_dim} --batch_size=${batch_size} --dimsort_range=${dimsort_range} --gridgcn_sample_opt=${gridgcn_sample_opt} --voxel_size=${voxel_size}
python test_semseg.py --log_dir pointnet2_sem_seg --test_area 5

# #Run sequential dimsort
# dimsort_range=16
# python setup.py --use_gpu --test_dimsort --presort --sort_dim=${sort_dim} --batch_size=${batch_size} --dimsort_range=${dimsort_range} --gridgcn_sample_opt=${gridgcn_sample_opt} --voxel_size=${voxel_size}
# python test_semseg.py --log_dir pointnet2_sem_seg --test_area 5

# #Run sequential dimsort
# dimsort_range=32
# python setup.py --use_gpu --test_dimsort --presort --sort_dim=${sort_dim} --batch_size=${batch_size} --dimsort_range=${dimsort_range} --gridgcn_sample_opt=${gridgcn_sample_opt} --voxel_size=${voxel_size}
# python test_semseg.py --log_dir pointnet2_sem_seg --test_area 5

# #Run sequential dimsort
# dimsort_range=64
# python setup.py --use_gpu --test_dimsort --presort --sort_dim=${sort_dim} --batch_size=${batch_size} --dimsort_range=${dimsort_range} --gridgcn_sample_opt=${gridgcn_sample_opt} --voxel_size=${voxel_size}
# python test_semseg.py --log_dir pointnet2_sem_seg --test_area 5

# #Run sequential dimsort
# dimsort_range=128
# python setup.py --use_gpu --test_dimsort --presort --sort_dim=${sort_dim} --batch_size=${batch_size} --dimsort_range=${dimsort_range} --gridgcn_sample_opt=${gridgcn_sample_opt} --voxel_size=${voxel_size}
# python test_semseg.py --log_dir pointnet2_sem_seg --test_area 5

# #Run parallel dimsort
# dimsort_range=2
# python setup.py --use_gpu --test_dimsort --parallel_option --presort --sort_dim=${sort_dim} --batch_size=${batch_size} --dimsort_range=${dimsort_range} --gridgcn_sample_opt=${gridgcn_sample_opt} --voxel_size=${voxel_size}
# python test_semseg.py --log_dir pointnet2_sem_seg --test_area 5

#Run parallel dimsort
dimsort_range=4
python setup.py --use_gpu --test_dimsort --parallel_option --presort --sort_dim=${sort_dim} --batch_size=${batch_size} --dimsort_range=${dimsort_range} --gridgcn_sample_opt=${gridgcn_sample_opt} --voxel_size=${voxel_size}
python test_semseg.py --log_dir pointnet2_sem_seg --test_area 5

# #Run parallel dimsort
# dimsort_range=8
# python setup.py --use_gpu --test_dimsort --parallel_option --presort --sort_dim=${sort_dim} --batch_size=${batch_size} --dimsort_range=${dimsort_range} --gridgcn_sample_opt=${gridgcn_sample_opt} --voxel_size=${voxel_size}
# python test_semseg.py --log_dir pointnet2_sem_seg --test_area 5

# #Run parallel dimsort
# dimsort_range=16
# python setup.py --use_gpu --test_dimsort --parallel_option --presort --sort_dim=${sort_dim} --batch_size=${batch_size} --dimsort_range=${dimsort_range} --gridgcn_sample_opt=${gridgcn_sample_opt} --voxel_size=${voxel_size}
# python test_semseg.py --log_dir pointnet2_sem_seg --test_area 5

# #Run parallel dimsort
# dimsort_range=32
# python setup.py --use_gpu --test_dimsort --parallel_option --presort --sort_dim=${sort_dim} --batch_size=${batch_size} --dimsort_range=${dimsort_range} --gridgcn_sample_opt=${gridgcn_sample_opt} --voxel_size=${voxel_size}
# python test_semseg.py --log_dir pointnet2_sem_seg --test_area 5

