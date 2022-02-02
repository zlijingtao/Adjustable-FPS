import torch
TEST_FPS = True

for i in range(B):
        seg_table = determine_segment(xyz[i, :, 2], num_segment)

        for j in range(S):

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
            # print(lower, upper)

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