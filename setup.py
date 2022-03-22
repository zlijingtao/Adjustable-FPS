#This script is used to change models/pointnet_util.py

import argparse

parser = argparse.ArgumentParser(description='PyTorch Dimsort Sampling Test')
parser.add_argument('--presort', action='store_true', default=False, help='set to true if PD data is presorted')
parser.add_argument('--parallel_option', action='store_true', default=False, help='set to true to enable parallel FPS')
parser.add_argument('--sort_dim', default=0, type=int, help='which axis the presorting is on?')
parser.add_argument('--save_computation_two_axis', action='store_true', default=False, help='set to true to only compute distnace on only two unsorted axises')
parser.add_argument('--visualize', action='store_true', default=False, help='set to true if you want visualization of a PD data of the sampling')
parser.add_argument('--use_gpu', action='store_true', default=False, help='if using gpu')
parser.add_argument('--batch_size', default=24, type=int, help='batch size of the PD data input, maximum is 24')
parser.add_argument('--test_dimsort', action='store_true', default=False, help='flag to enable dimsort heuristic')
parser.add_argument('--dimsort_range', default=4, type=int, help='specify the dimsort range')
parser.add_argument('--test_grid_gcn', action='store_true', default=False, help='flag to enable grid gcn')
parser.add_argument('--gridgcn_sample_opt', default='rvs', type=str, help='grid-gcn sample option, choose from cas or rvs')
parser.add_argument('--voxel_size', default=40, type=int, help='voxel_size for grid-gcn')
args = parser.parse_args()


#Do sanity check

if args.gridgcn_sample_opt != "rvs" and args.gridgcn_sample_opt != "cas":
    raise("grid-gcn sample option is wrong, choose from cas or rvs!")

if args.batch_size > 24:
    print("batch size surpasses the maximum batch size of 24 (of fixed data input we are using)")

#copy in-place to pointnet_util.py
file_name = "./models/pointnet_util.py"

with open(file_name, "r") as in_file:
        buf = in_file.readlines()

with open(file_name, "w") as out_file:
    for line in buf:
        try:
            if "PRESORT_FLAG = " in line:
                line = "PRESORT_FLAG = True\n" if args.presort else "PRESORT_FLAG = False\n"
            elif "PARALLEL_OPTION = " in line:
                line = "PARALLEL_OPTION = True\n" if args.parallel_option else "PARALLEL_OPTION = False\n"
            elif "SELECT_DIM = " in line:
                line = "SELECT_DIM = {}\n".format(args.sort_dim)
            elif "SAVE_COMPUTATION_TWO_AXIS = " in line:
                line = "SAVE_COMPUTATION_TWO_AXIS = True\n" if args.save_computation_two_axis else "SAVE_COMPUTATION_TWO_AXIS = False\n"

            elif "VISUALIZE = " in line:
                line = "VISUALIZE = True\n" if args.visualize else "VISUALIZE = False\n"
            elif "USE_GPU = " in line:
                line = "USE_GPU = True\n" if args.use_gpu else "USE_GPU = False\n"
            elif "BATCH_SIZE = " in line:
                line = "BATCH_SIZE = {}\n".format(args.batch_size)

            elif "TEST_DIMSORT = " in line:
                line = "TEST_DIMSORT = True\n" if args.test_dimsort else "TEST_DIMSORT = False\n"
            elif "DIMSORT_RANGE = " in line:
                line = "DIMSORT_RANGE = {}\n".format(args.dimsort_range)

            elif "TEST_GRIDGCN = " in line:
                line = "TEST_GRIDGCN = True\n" if args.test_grid_gcn else "TEST_GRIDGCN = False\n"
            elif "GRIDGCN_SAMPLE_OPT = " in line:
                line = "GRIDGCN_SAMPLE_OPT = \"{}\"\n".format(args.gridgcn_sample_opt)
            elif "VOXEL_SIZE = " in line:
                line = "VOXEL_SIZE = {}\n".format(args.voxel_size)

            out_file.write(line)
        except:
            out_file.write(line)