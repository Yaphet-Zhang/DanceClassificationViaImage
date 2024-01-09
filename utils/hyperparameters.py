# -*- coding: utf-8 -*-
# @Time    : 2023-03-02 11:00
# @Author  : zhangbowen


########## common ##########
BATCH_SIZE = 2
EPOCH = 50
VAL_DATA = 0.2
NAME_CLASSES= ['no-dance','dance']


########## openpose ##########
INPUT_SIZE_1 = (368, 368)
COLOR_MEAN = [0.485, 0.456, 0.406]
COLOR_STD = [0.229, 0.224, 0.225]




########## lstm ##########
# feature number / input size
num_feature = 2 # x coord, y coord 
num_joint = 17
INPUT_SIZE = num_feature*num_joint

# time step number / sequence size
num_frame = 1
num_second = 4
SEQUENCE_SIZE = num_frame*num_second

HIDDEN_SIZE = 32 # customize
NUM_LAYERS = 1 # number of RNN for stacking
OUTPUT_SIZE = 2
NAME_NET = 'lstm'


########## gru ##########
# # sum number of features = INPUT_SIZE * SEQUENCE_SIZE
# INPUT_SIZE = 8
# SEQUENCE_SIZE = 17
# HIDDEN_SIZE = 32 # customize
# NUM_LAYERS = 2 # number of RNN for stacking
# OUTPUT_SIZE = 2
# NAME_NET = 'gru'


