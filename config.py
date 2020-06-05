# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 17:22:57 2018

@author: LongJun
"""
import os
import numpy as np

dataset = 'dataset'

pkl = 'pkl'

batch_size = 1

# CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
#            'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
#            'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
#            'train', 'tvmonitor']

CLASSES = ['background', '1', '2', '3', '4', '5']

#max and min img size
target_size = 1024

max_size = 1024

#FLIPPED = False

#max training step
MAX_ITER = 214800 #90000 for VOC2007+VOC2012, 70000 for VOC2007

#the step of LEARNING_RATE decay
lr_change_ITER = 214800*0.7 #70000 for VOC2007+VOC2012 50000 for VOC2007

LEARNING_RATE = [0.001, 0.0001]

SUMMARY_ITER = 2148

SAVE_ITER = 2148

#threshold for anchor label
overlaps_max = 0.7

overlaps_min = 0.3

CKPT = os.path.join('ckpt')

Summary = 'summary'

momentum = 0.9

anchor_scales = [128,256,512]

anchor_ratios = [0.5,1,2]

anchor_batch = 256

weight_path = os.path.join('vgg_16', 'vgg_16.ckpt')

weigt_output_path = CKPT

test_output_path = 'results'

feat_stride = 16

PIXEL_MEANS = np.array([[[122.7717, 115.9465, 102.9801]]])

#roi nms config 
max_rpn_input_num = 12000

max_rpn_nms_num = 2000

test_max_rpn_input_num = 6000

test_max_rpn_nms_num = 300

nms_thresh = 0.7

#batch for dection network
dect_train_batch = 256

dect_fg_rate = 0.25

bbox_nor_target_pre = True

bbox_nor_mean = (0.0, 0.0, 0.0, 0.0)

bbox_nor_stdv = (0.1, 0.1, 0.2, 0.2)

roi_input_inside_weight = (1.0, 1.0, 1.0, 1.0)

POOLING_SIZE = 7

#threshold for roi
fg_thresh = 0.5

bg_thresh_hi = 0.5

bg_thresh_lo = 0.0

test_nms_thresh = 0.3

test_fp_tp_thresh = 0.5

test_max_per_image = 100

#test_image_show num
img_save_num = 2148

image_output_dir = os.path.join(test_output_path)
