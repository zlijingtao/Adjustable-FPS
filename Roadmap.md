# Pytorch Implementation of PointNet and PointNet++ 

This repo is implementation for [PointNet](http://openaccess.thecvf.com/content_cvpr_2017/papers/Qi_PointNet_Deep_Learning_CVPR_2017_paper.pdf) and [PointNet++](http://papers.nips.cc/paper/7095-pointnet-deep-hierarchical-feature-learning-on-point-sets-in-a-metric-space.pdf) in pytorch.

## Roadmap
**2022/01/03:**

(1) Optimize the speed of QB kernel (torch-lize). That this can be also used for general platforms such as CPU and GPU.

(2) Added testing codes, including classification and segmentation, and semantic segmentation with visualization. 

### Setup (Edit in models/pointnet_util.py)
```
DIMSORT = True ## Set to enable DIMSORT

DIMSORT_DIV = 16 ## Set the fixed number (absolute points want to pay attention).

DIMSORT_RANGE = int(2048 / DIMSORT_DIV)

VISUALIZE = True

TEST_FPS = False ## Test DIMSORT-FPS by disabling DIMSORT-QB

TEST_QB = False ## Test DIMSORT-QB by disabling DIMSORT-FPS
```

### Run
```
## Check model in ./models 
## E.g. pointnet2_msg
python test_partseg.py --normal --log_dir pointnet2_part_seg_msg
```

### Performance
| Model | Accuracy(%) / avg IoU | Latency |
|--|--|--|
|Vanilla|	**83.3**|0.158s/0.031s|
|DIMSORT-1024|	**83.0**| |
|DIMSORT-512|	**82.8**| |
|DIMSORT-256|	**82.8**| |
|DIMSORT-128|	**82.5**| |
|DIMSORT-64|	**83.0**| |
|DIMSORT-32|	**91.6/82.6**| |
|DIMSORT-16|	**91.1/82.0**| 0.505s/0.091s|
|DIMSORT-8|	**91.2, 91.0, 91.4/82.2, 81.7, 82.4**| 0.502s/0.095s|
|DIMSORT-4|	**76.6, 76.5, 75.5/61.9, 61.7, 60.8**| 0.492s/0.093s|
|Grid-GCN-40 (RVS)|	**91.0, 91.0, 91.2. 90.6/82.3, 82.1, 82.7, 82.5**| 2.12s/1.68s|
|Grid-GCN-40 (CAS)|	**91.3, 91.0, 91.3/81.5, 81.2, 82.8**| 3.06s/2.22s|