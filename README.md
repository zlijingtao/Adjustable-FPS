# Adjustable Farthest Point Sampling Method for PointNet++

This repo is adapted from [
Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch), for testing the a heuristic Sampling algorithm Performance on downstreaming PointNet++ tasks. This work is published as proceedings of SIPS 2022 and its pdf is available at (IEEE, final version)[https://ieeexplore.ieee.org/abstract/document/9919246] and (Arxiv, accepted version)[https://arxiv.org/pdf/2208.08795.pdf]

### Run

First, please refer to Pointnet_Pointnet2_pytorch and train a pointnet++ model on three tasks.

Then, to play with the proposed sample methods, please change arguments and add necessary flags to setup.py call in bash files, for example:

```
python setup.py --use_gpu --parallel_option --presort --coarse_sort_k=${coarse_sort_k} --parallel_m=${parallel_m} --sort_dim=${sort_dim} --batch_size=${batch_size} --dimsort_range=${dimsort_range} --gridgcn_sample_opt=${gridgcn_sample_opt} --voxel_size=${voxel_size}
```

Finally, test its performance using the pre-trained pointnet++ model.

```
python test_partseg.py --normal --log_dir pointnet2_part_seg_msg
```

<!-- ### Run
```
bash run_inference.sh
```

### Performance Recordings (for part-segmentation only)

| Model | Accuracy(%) | Avg IoU | Latency (not rigorous) |
|--|--|--|--|
|**Seq-Vanilla**|0.91308 (0.00136)|0.82808 (0.00471)|12.61348 (0.23503)|
|**Seq-Vanilla (save_comp)**|0.90486 (0.00147)|0.81706 (0.00353)|12.21645 (0.08228)|
|Parallel-Vanilla (sorted)|0.91349 (0.00247)|0.82662 (0.00415)|12.0452 (0.10663)|
|Parallel-Vanilla (no-sort)|0.90630 (0.00164)|0.81049 (0.00546)|12.25779 (0.08752)|
|Parallel-DIMSORT-8|0.91120 (0.00269)|0.82119 (0.00367)| 12.41202 (0.13205)|
|Parallel-DIMSORT-8 (save_comp)|0.91071 (0.00247) |0.82081 (0.00289)| 12.13942 (0.10859)|
|Parallel-DIMSORT-8 (no-sort)|0.90972 (0.00268)|0.81659 (0.00479)|12.51211 (0.10661)|
|Parallel-DIMSORT-8 (no-sort, save_comp)|0.91101 (0.00225)|0.81700 (0.00641)|12.33169 (0.09527)|
|Parallel-DIMSORT-4|0.91149 (0.00182)|0.82006 (0.00351)| 12.26777 (0.10436)|
|**Parallel-DIMSORT-4 (save_comp)**|0.91126 (0.00296)|0.81992 (0.00345)| 12.10241 (0.07481)|
|Parallel-DIMSORT-4 (no-sort)|0.91179 (0.00281)|0.81678 (0.00576)|12.38499 (0.10152)|
|Parallel-DIMSORT-4 (no-sort, save_comp)|0.91090 (0.00245)|0.81688 (0.00690)|12.15143 (0.06635)|
|Parallel-DIMSORT-2|0.90950 (0.00103)|0.81316 (0.00607)| 11.95670 (0.08510)|
|Parallel-DIMSORT-2 (save_comp)|0.90861 (0.00201)|0.80909 (0.00577)| 11.95530 (0.11242)|
|Parallel-DIMSORT-2 (no-sort)|0.91022 (0.00248)|0.81432 (0.00499)|12.42760 (0.07444)|
|Seq-DIMSORT-8|	0.90863 (0.00225)|0.81391 (0.00596)| 12.36809 (0.04590)|
|Seq-DIMSORT-8 (save_comp)|	0.90948 (0.00268)|0.81608 (0.00690)| 12.16529 (0.10634)|
|Seq-DIMSORT-8 (no-sort)|	0.90994 (0.00280)|0.81451 (0.00622)| 12.58202 (0.09947)|
|**Grid-GCN-40 (RVS)**|0.91110 (0.00143)|0.82380 (0.00422)| 13.46429 (0.10773)|
|**Grid-GCN-40 (CAS)**|0.90940 (0.00165)|0.82015 (0.00560)| 17.53124 (0.15838)|

(parallel is default set to 16-core) -->