"""
Author: Benny
Date: Nov 2019
"""
from data_utils.ModelNetDataLoader import ModelNetDataLoader
import argparse
import numpy as np
from models.pointnet_util import pcloud_sort, PRESORT_FLAG, VISUALIZE, SELECT_DIM, USE_GPU, BATCH_SIZE
import os
import torch
import logging
from PIL import Image
from visualizer.pc_utils import point_cloud_three_views
from tqdm import tqdm
import sys
import importlib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
    parser.add_argument('--log_dir', type=str, default='pointnet2_ssg_normal', help='Experiment root')
    parser.add_argument('--normal', action='store_true', default=True, help='Whether to use normal information [default: False]')
    parser.add_argument('--num_votes', type=int, default=3, help='Aggregate classification scores with voting [default: 3]')
    parser.add_argument('--num_batch', type=int, default=103, help='Number of batch to do the inference')
    return parser.parse_args()

def test(model, loader, num_class=40, vote_num=1):
    mean_correct = []
    class_acc = np.zeros((num_class,3))
    for j, data in tqdm(enumerate(loader), total=len(loader)):
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        torch.save(points, "class_test_sample/class_point_batch_{}.pt".format(j))
        torch.save(target, "class_test_sample/class_target_batch_{}.pt".format(j))
        if USE_GPU:
            points, target = points.cuda(), target.cuda()
        classifier = model.eval()
        vote_pool = torch.zeros(target.size()[0],num_class)
        if USE_GPU:
            vote_pool = vote_pool.cuda()
        for _ in range(vote_num):
            pred, _ = classifier(points)
            vote_pool += pred
        pred = vote_pool/vote_num
        pred_choice = pred.data.max(1)[1]
        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target==cat].eq(target[target==cat].long().data).cpu().sum()
            class_acc[cat,0]+= classacc.item()/float(points[target==cat].size()[0])
            class_acc[cat,1]+=1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item()/float(points.size()[0]))
    class_acc[:,2] =  class_acc[:,0]/ class_acc[:,1]
    class_acc = np.mean(class_acc[:,2])
    instance_acc = np.mean(mean_correct)
    return instance_acc, class_acc

def single_test(model, num_batch, num_class=40, vote_num=1):
    global VISUALIZE
    mean_correct = []
    class_acc = np.zeros((num_class,3))
    if num_batch > 103:
        num_batch = 103
    for j in tqdm(range(num_batch), total=num_batch):
        points = torch.load("class_test_sample//class_point_batch_{}.pt".format(j))
        target = torch.load("class_test_sample//class_target_batch_{}.pt".format(j))
        
        if BATCH_SIZE < 24:
            points = points[:BATCH_SIZE, :, :]
            target = target[:BATCH_SIZE, :]

        if USE_GPU:
            points, target = points.cuda(), target.cuda()
    

        
        if PRESORT_FLAG:
            points = points.permute(0, 2, 1)
            points, target = pcloud_sort(points, target, sel_dim = SELECT_DIM)
            points = points.permute(0, 2, 1)
        if VISUALIZE and j == 0:
            # print("save original PD")
            im_array = point_cloud_three_views(points.permute(0, 2, 1).cpu().numpy()[0, :, :])
            img = Image.fromarray(np.uint8(im_array * 255.0))
            img.save('pd0-orig.jpg')
            VISUALIZE = False
        classifier = model.eval()
        vote_pool = torch.zeros(target.size()[0],num_class)
        if USE_GPU:
            vote_pool = vote_pool.cuda()
        for _ in range(vote_num):
            pred, _ = classifier(points)
            vote_pool += pred
        pred = vote_pool/vote_num
        pred_choice = pred.data.max(1)[1]
        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target==cat].eq(target[target==cat].long().data).cpu().sum()
            class_acc[cat,0]+= classacc.item()/float(points[target==cat].size()[0])
            class_acc[cat,1]+=1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item()/float(points.size()[0]))

    class_acc[:,2] =  class_acc[:,0]/ class_acc[:,1]
    class_acc = np.mean(class_acc[:,2])
    instance_acc = np.mean(mean_correct)
    return instance_acc, class_acc


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    use_gpu = USE_GPU
    '''HYPER PARAMETER'''
    if use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    experiment_dir = 'log/classification/' + args.log_dir

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

    # '''DATA LOADING'''
    # log_string('Load dataset ...')
    # DATA_PATH = 'data/modelnet40_normal_resampled/'
    # TEST_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_point, split='test', normal_channel=args.normal)
    # testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=4)

    '''MODEL LOADING'''
    num_class = 40
    model_name = os.listdir(experiment_dir+'/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    if USE_GPU:
        classifier = MODEL.get_model(num_class,normal_channel=args.normal).cuda()
    else:
        classifier = MODEL.get_model(num_class,normal_channel=args.normal)

    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])

    with torch.no_grad():
        # instance_acc, class_acc = test(classifier.eval(), testDataLoader, vote_num=args.num_votes)
        instance_acc, class_acc = single_test(classifier.eval(), args.num_batch, vote_num=args.num_votes)
        log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))



if __name__ == '__main__':
    args = parse_args()
    main(args)
