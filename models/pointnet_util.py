import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np
from PIL import Image
from visualizer.pc_utils import point_cloud_three_views
DIMSORT = True
DIMSORT_DIV = 16
DIMSORT_RANGE = int(2048 / DIMSORT_DIV)
VISUALIZE = True
TEST_FPS = False
TEST_QB = False
def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def index_points_query_ball(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape

    # """TODO: Accelerate this"""
    group_idx = torch.zeros([B, S, nsample]).long()
    permu_list = torch.randperm(N)
    for i in range(B):
        for j in range(S):
            count = 0
            for k in permu_list:
                dist = (torch.sum(torch.square(xyz[i, k, :] - new_xyz[i, j, :])))
                if dist < radius ** 2:
                    group_idx[i, j, count] = k
                    count += 1
                    if count >= nsample:
                        break
            while count <= nsample - 1:
                group_idx[i, j, count] = group_idx[i, j, count - 1]
                count += 1
    
    new_points = index_points(xyz, group_idx)

    return new_points

def determine_segment(sorted_array, num_segment):
    seg_table = [0]
    length = len(sorted_array)
    level_gap = 2.0 / num_segment
    current_level = -1.0 + level_gap
    for i in range(length):
        if sorted_array[i] > current_level:
            seg_table.append(i)
            current_level += level_gap
    while len(seg_table) < num_segment + 1:
        seg_table.append(length)
    return seg_table

def index_points_query_ball_include_points(radius, nsample, xyz, xyz2, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape

    # """TODO: Accelerate this"""
    group_idx = torch.zeros([B, S, nsample]).long()
    permu_list = torch.randperm(N)
    num_segment = 2.0 // radius
    if num_segment > DIMSORT_DIV:
        num_segment = DIMSORT_DIV
    actual_computation = 0
    for i in range(B):
        if not TEST_FPS:
            seg_table = determine_segment(xyz[i, :, 2], num_segment)
            # print(seg_table)

        for j in range(S):

            if not TEST_FPS:
                seg_pos = int(((new_xyz[i, j, 2] + 1.0) * num_segment) // 2)
                if seg_pos > 0 and seg_pos < len(seg_table) - 2:
                    lower = seg_table[seg_pos - 1]
                    upper = seg_table[seg_pos + 2]
                elif seg_pos > 0:
                    lower = seg_table[seg_pos - 1]
                    upper = seg_table[-1]
                elif seg_pos < len(seg_table) - 2:
                    lower = seg_table[0]
                    upper = seg_table[seg_pos + 2]
                else:
                    lower = seg_table[0]
                    upper = seg_table[-1]
                actual_computation += upper - lower
                
                permu_list = lower + torch.randperm(upper - lower)

            count = 0
            for k in permu_list:
                dist = (torch.sum(torch.square(xyz[i, k, :] - new_xyz[i, j, :])))
                if dist < radius ** 2:
                    group_idx[i, j, count] = k
                    count += 1
                    if count >= nsample:
                        break
            while count <= nsample - 1:
                group_idx[i, j, count] = group_idx[i, j, count - 1]
                count += 1
    if not TEST_FPS:
        print("Dimsort's QB computation is {:2.2f}% of the original, ({}/ {})".format(100 * actual_computation / (B * S * N), actual_computation, B * S * N))
    new_points = index_points(xyz, group_idx)
    new_points2 = index_points(xyz2, group_idx)

    return new_points, new_points2

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    actual_computation = 0
    if DIMSORT and not TEST_QB:
        for b in range(B):
            distance = torch.ones(1, N).to(device) * 1e10
            farthest = torch.randint(0, N, (1,), dtype=torch.long).to(device)
            for i in range(npoint):
                centroids[b, i] = farthest
                centroid = xyz[b, farthest, :].view(1, 3)
                upper = farthest + DIMSORT_RANGE//2 if farthest <= N - DIMSORT_RANGE//2 else N
                lower = farthest - DIMSORT_RANGE//2 if farthest > DIMSORT_RANGE//2 else 0
                dist = torch.ones(1, N).to(device) * 1e10
                dist[0, lower:upper] = torch.sum((xyz[b, lower:upper, :] - centroid) ** 2, -1)
                actual_computation += (upper - lower)
                mask = dist < distance
                distance[mask] = dist[mask]
                farthest = torch.max(distance, dim = -1)[1].long()
        print("Dimsort's FPS computation is {:2.2f}% of the original, ({}/{})".format(100 * int(actual_computation) / (B * npoint * N), int(actual_computation), B * npoint * N))
    else:
        distance = torch.ones(B, N).to(device) * 1e10
        farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
        batch_indices = torch.arange(B, dtype=torch.long).to(device)
        for i in range(npoint):
            centroids[:, i] = farthest
            centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
            dist = torch.sum((xyz - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, -1)[1]
    
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])

    sqrdists = square_distance(new_xyz, xyz) # calculate square distance in a sequential way and xxx

    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def pcloud_sort(npoint, npoint2 = None, sel_dim = -1):
    #firstly choose the longest dimension to sort
    batch_size = npoint.size(0)
    if sel_dim == -1:
        range_x = torch.max(npoint[:, :, 0]) - torch.min(npoint[:, :, 0])
        range_y = torch.max(npoint[:, :, 1]) - torch.min(npoint[:, :, 1])
        range_z = torch.max(npoint[:, :, 2]) - torch.min(npoint[:, :, 2])
        if range_x >= range_y and range_x >= range_z:
            sel_dim = 0
        elif range_y >= range_z:
            sel_dim = 1
        else:
            sel_dim = 2
    output = torch.zeros_like(npoint)
    if npoint2 is not None:
        output2 = torch.zeros_like(npoint2)
    for i in range(batch_size):
        _, idx = torch.sort(npoint[i, :, sel_dim])
        output[i, :, :] = npoint[i, idx, :]
        if npoint2 is not None:
            if len(output2.size()) == 3:
                output2[i, :, :] = npoint2[i, idx, :]
            elif len(output2.size()) == 2:
                output2[i, :] = npoint2[i, idx]
            
    if npoint2 is not None:
        return output, output2
    else:
        return output

def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    torch.cuda.empty_cache()
    new_xyz = index_points(xyz, fps_idx)
    torch.cuda.empty_cache()
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    torch.cuda.empty_cache()
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    torch.cuda.empty_cache()
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    torch.cuda.empty_cache()

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint

        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        
        if VISUALIZE:
            print("save FPS sampled PD")
            im_array = point_cloud_three_views(new_xyz.numpy()[0, :, :])
            img = Image.fromarray(np.uint8(im_array * 255.0))
            img.save('pd0-sample{}.jpg'.format(new_xyz.size(1)))
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            
            if points is not None:
                if DIMSORT:
                    grouped_xyz, grouped_points = index_points_query_ball_include_points(radius, K, xyz, points, new_xyz)
                    grouped_xyz -= new_xyz.view(B, S, 1, C)
                    grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
                else:
                    group_idx = query_ball_point(radius, K, xyz, new_xyz)
                    grouped_xyz = index_points(xyz, group_idx) 
                    grouped_points = index_points(points, group_idx)
                    grouped_xyz -= new_xyz.view(B, S, 1, C)
                    grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1) 
            else:
                if DIMSORT:
                    grouped_xyz = index_points_query_ball(radius, K, xyz, new_xyz)
                else:
                    group_idx = query_ball_point(radius, K, xyz, new_xyz)
                    grouped_xyz = index_points(xyz, group_idx) 
                grouped_xyz -= new_xyz.view(B, S, 1, C)
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points

