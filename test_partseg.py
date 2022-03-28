"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
# from data_utils.ShapeNetDataLoader import PartNormalDataset
import torch
import logging
import sys
import importlib
from tqdm import tqdm
import numpy as np
from models.pointnet_util import pcloud_sort, PRESORT_FLAG, VISUALIZE, SELECT_DIM, USE_GPU, BATCH_SIZE
from PIL import Image
from visualizer.pc_utils import point_cloud_three_views
from time import time
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43], 'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
seg_label_to_cat = {} # {0:Airplane, 1:Airplane, ...49:Table}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in testing [default: 24]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device [default: 0]')
    parser.add_argument('--num_point', type=int, default=2048, help='Point Number [default: 2048]')
    parser.add_argument('--log_dir', type=str, default='pointnet2_part_seg_ssg', help='Experiment root')
    parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')
    parser.add_argument('--num_votes', type=int, default=1, help='Aggregate segmentation scores with voting [default: 3]') # Simplify the process
    parser.add_argument('--num_batch', type=int, default=120, help='Number of batch to do the inference')
    parser.add_argument('--avg_time', type=int, default=10, help='Number of average to do the inference')
    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    use_gpu = USE_GPU
    global VISUALIZE
    '''HYPER PARAMETER'''
    if use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu       

    experiment_dir = 'log/part_seg/' + args.log_dir

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    num_classes = 16
    num_part = 50

    '''MODEL LOADING'''
    model_name = os.listdir(experiment_dir+'/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    if use_gpu:
        classifier = MODEL.get_model(num_part, normal_channel=args.normal).cuda()
    else:
        classifier = MODEL.get_model(num_part, normal_channel=args.normal)
    if use_gpu:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    else:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth', map_location=torch.device('cpu'))
    classifier.load_state_dict(checkpoint['model_state_dict'])

    accu_list = []
    mIoU_list = []
    timecost_list = []
    with torch.no_grad():

        for _ in range(args.avg_time):
            test_metrics = {}
            total_correct = 0
            total_seen = 0
            total_seen_class = [0 for _ in range(num_part)]
            total_correct_class = [0 for _ in range(num_part)]
            shape_ious = {cat: [] for cat in seg_classes.keys()}
            seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
            for cat in seg_classes.keys():
                for label in seg_classes[cat]:
                    seg_label_to_cat[label] = cat
            classifier = classifier.eval()
            if args.num_batch > 120:
                args.num_batch = 120
            for j in tqdm(range(args.num_batch), total=args.num_batch):
                
                points = torch.load("partseg_test_sample/partseg_point_batch_{}.pt".format(j))
                label = torch.load("partseg_test_sample/partseg_label_batch_{}.pt".format(j))
                target = torch.load("partseg_test_sample/partseg_target_batch_{}.pt".format(j))
                
                cur_batch_size, NUM_POINT, _ = points.transpose(2, 1).size()
                if USE_GPU:
                    points, label, target = points.cuda(), label.cuda(), target.cuda()

                if BATCH_SIZE < 24:
                    points = points[:BATCH_SIZE, :, :]
                    # print(points.size())
                    label = label[:BATCH_SIZE]
                    # print(label.size())
                    target = target[:BATCH_SIZE, :]
                
                if PRESORT_FLAG:
                    # print(points.size())
                    # print(target.size())
                    # points = points.transpose(2, 1)
                    points, target = pcloud_sort(points.transpose(2, 1), target, sel_dim = SELECT_DIM)
                    points = points.transpose(2, 1)
                
                if VISUALIZE and j == 0:
                    # print("save original PD")
                    im_array = point_cloud_three_views(points.numpy()[0, :, :])
                    img = Image.fromarray(np.uint8(im_array * 255.0))
                    img.save('pd0-orig.jpg')
                    VISUALIZE = False

                

                if use_gpu:
                    vote_pool = torch.zeros(target.size()[0], target.size()[1], num_part).cuda()
                else:
                    vote_pool = torch.zeros(target.size()[0], target.size()[1], num_part)
                start_time = time()
                for _ in range(args.num_votes):
                    seg_pred, _ = classifier(points, to_categorical(label, num_classes))
                    vote_pool += seg_pred
                
                time_cost = time() - start_time
                seg_pred = vote_pool / args.num_votes
                cur_pred_val = seg_pred.cpu().data.numpy()
                cur_pred_val_logits = cur_pred_val
                cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
                target = target.cpu().data.numpy()
                for i in range(cur_batch_size):
                    cat = seg_label_to_cat[target[i, 0]]
                    logits = cur_pred_val_logits[i, :, :]
                    cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]
                correct = np.sum(cur_pred_val == target)
                total_correct += correct
                total_seen += (cur_batch_size * NUM_POINT)

                for l in range(num_part):
                    total_seen_class[l] += np.sum(target == l)
                    total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))

                for i in range(cur_batch_size):
                    segp = cur_pred_val[i, :]
                    segl = target[i, :]
                    cat = seg_label_to_cat[segl[0]]
                    part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
                    for l in seg_classes[cat]:
                        if (np.sum(segl == l) == 0) and (
                                np.sum(segp == l) == 0):  # part is not present, no prediction as well
                            part_ious[l - seg_classes[cat][0]] = 1.0
                        else:
                            part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                                np.sum((segl == l) | (segp == l)))
                    shape_ious[cat].append(np.mean(part_ious))
            all_shape_ious = []
            for cat in shape_ious.keys():
                for iou in shape_ious[cat]:
                    all_shape_ious.append(iou)
                shape_ious[cat] = np.mean(shape_ious[cat])
            mean_shape_ious = np.mean(list(shape_ious.values()))
            test_metrics['accuracy'] = total_correct / float(total_seen)
            test_metrics['class_avg_accuracy'] = np.mean(
                np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))
            # for cat in sorted(shape_ious.keys()):
            #     log_string('eval mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))
            test_metrics['class_avg_iou'] = mean_shape_ious
            test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)
            
        accu_list.append(test_metrics['accuracy'])
        mIoU_list.append(test_metrics['inctance_avg_iou'])
        timecost_list.append(time_cost)

    log_string('Accuracy (10-time avg/std) is: %.5f (%.5f)'%(np.average(accu_list), np.std(accu_list)))
    log_string('Class avg accuracy is: %.5f'%test_metrics['class_avg_accuracy'])
    log_string('Class avg mIOU is: %.5f'%test_metrics['class_avg_iou'])
    log_string('Inctance avg mIOU (10-time avg) is: %.5f (%.5f)'%(np.average(mIoU_list), np.std(mIoU_list)))
    log_string('Time cost (10-time avg) is: %.5f (%.5f)'%(np.average(timecost_list), np.std(timecost_list)))

if __name__ == '__main__':
    args = parse_args()
    main(args)

