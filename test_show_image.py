# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 09:54:12 2018

@author: LongJun
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import config as cfg
import os
import pascal_voc as pascl
import numpy as np
import tensorflow as tf
import network
import cv2
from nms import py_cpu_nms

class Val_test(object):   
    def __init__(self, net ,val_data):
        self.net = net
        self.val_data = val_data
        self.ckpt_filename = tf.train.latest_checkpoint(cfg.CKPT)
        self.test_output_dir = cfg.test_output_path
        self.image_output_dir = cfg.image_output_dir



    def test_model(self):
        saver = tf.train.Saver()
        _rois_coord = self.net.rois_coord[:,1:5]
        _pred_box = self.net.bbox_pred
        _pred_score = self.net.cls_prob
        _pred_box_score_arg = tf.argmax(_pred_score, axis=1)
        dect_total_result = [[[] for i in range(cfg.img_save_num)] for j in range(self.net.num_classes)]
        test_data = [[] for i in range (cfg.img_save_num)]
        with tf.Session() as sess:
            saver.restore(sess, self.ckpt_filename)
            for i in range (cfg.img_save_num):
                print (i, ' image test compeleted')
                train_data = self.val_data.get()  #if you want to change the test image, you can using cv2.imread() here to read your own image data
                test_data[i] = train_data
                image = cv2.imread(test_data[i]['imname'])
                image_height = np.array(train_data['image'].shape[1])
                image_width = np.array(train_data['image'].shape[2])
                feed_dict = {self.net.image: train_data['image'], self.net.image_width: image_width,\
                             self.net.image_height: image_height}
                                
                rois_coord, pred_box, pred_box_score_arg, pred_score= sess.run([_rois_coord, _pred_box, _pred_box_score_arg, _pred_score],\
                                                                        feed_dict=feed_dict)
                pred_box_score_arg = pred_box_score_arg.astype(np.int32)
                num_pred = pred_box_score_arg.shape[0]
                pred_box_gather = np.empty([num_pred, 4], dtype = np.float32)
                pred_score_gather = np.empty(num_pred)
                
                for j in range(num_pred):
                    pred_box_gather[j, :] = pred_box[j, 4*pred_box_score_arg[j]:4*(pred_box_score_arg[j]+1)]
                    pred_score_gather[j] = pred_score[j, pred_box_score_arg[j]]
                    
                pred_box_gather = pred_box_gather * np.array(cfg.bbox_nor_stdv) + np.array(cfg.bbox_nor_mean)
                pre_box_coord = self.coord_transform_inv(rois_coord, pred_box_gather.astype(np.float32))
                pre_box_coord = pre_box_coord
                for k in range(1, self.net.num_classes):
                    pre_class_arg = np.where(pred_box_score_arg==k)[0]
                    cls_pred_box_coord = pre_box_coord[pre_class_arg, :]
                    cls_pred_score = pred_score_gather[pre_class_arg]
                    #print(cls_pred_box_coord.shape, cls_pred_score.shape)
                    cls_pred_score = cls_pred_score[:, np.newaxis]  
                    cls_pred_target = np.concatenate((cls_pred_box_coord, cls_pred_score), axis=1)
                    keep = py_cpu_nms(cls_pred_target, cfg.test_nms_thresh)
                    cls_pred_target = cls_pred_target[keep, :]

                    for h in range(len(cls_pred_target)):
                        if cls_pred_target[h][4] > 0.5:
                            x1 = int(cls_pred_target[h][0])
                            y1 = int(cls_pred_target[h][1])
                            x2 = int(cls_pred_target[h][2])
                            y2 = int(cls_pred_target[h][3])
                            cv2.rectangle(image, (x1, y1), (x2, y2), ((k % 19) * 15, (k % 3) * 100 + 40, (k % 6) * 50 + 35),
                                          2)
                            cv2.rectangle(image, (x1, y1 - 20), (x2, y1), (125, 125, 125), -1)
                            lineType = cv2.LINE_AA if cv2.__version__ > '3' else cv2.CV_AA
                            cv2.putText(image, cfg.CLASSES[k] + ' : %.2f' % cls_pred_target[h][4], \
                                        (x1 + 5, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, \
                                        (0, 0, 0), 1, lineType)
                if not os.path.exists(self.image_output_dir):
                    os.mkdir(self.image_output_dir)
                im_save_path = os.path.join(self.image_output_dir, test_data[i]['imname'].split('/')[-1])
                cv2.imwrite(im_save_path, image)
                cv2.imshow('Image', image)
                cv2.waitKey(0)

            for k in range(cfg.img_save_num):
                imname = test_data[k]['imname']
                im_scale = test_data[k]['scale']
                image = cv2.imread(imname)
                im = cv2.resize(image, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
                im = self.draw_result(im, dect_total_result, k)
                if not os.path.exists(self.image_output_dir):
                    os.mkdir(self.image_output_dir)
                im_save_path = os.path.join(self.image_output_dir, imname.split('/')[-1])
                cv2.imwrite(im_save_path, im)
                cv2.imshow('Image',im)
                cv2.waitKey(0)
        
        
    def coord_transform_inv (self, anchors, boxes):
        anchors = anchors.astype(np.float32)
        anchors = np.reshape(anchors, [-1,4])
        anchor_x = (anchors[:,2] + anchors[:,0]) * 0.5
        anchor_y = (anchors[:,3] + anchors[:,1]) * 0.5
        acnhor_w = (anchors[:,2] - anchors[:,0]) + 1.0
        acnhor_h = (anchors[:,3] - anchors[:,1]) + 1.0
        boxes = np.reshape(boxes, [-1,4])
        boxes_x = boxes[:,0]*acnhor_w + anchor_x
        boxes_y = boxes[:,1]*acnhor_h + anchor_y
        boxes_w = np.exp(boxes[:,2])*acnhor_w
        boxes_h = np.exp(boxes[:,3])*acnhor_h
        coord_x1 = boxes_x - boxes_w*0.5
        coord_y1 = boxes_y - boxes_h*0.5
        coord_x2 = boxes_x + boxes_w*0.5
        coord_y2 = boxes_y + boxes_h*0.5
        coord_result = np.stack([coord_x1, coord_y1, coord_x2, coord_y2], axis=1)
        return coord_result
    
    

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    net = network.Net(is_training=False)
    val_data = pascl.pascal_voc('test', fliped=False)
    test = Val_test(net, val_data)
    print ('start training')
    test.test_model()
