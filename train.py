import os
import network
import numpy as np
import config as cfg
import tensorflow as tf
import pascal_voc as pascl
from losslayer import RPN_loss
from predict_loss import Predict_loss
from tensorflow.python import pywrap_tensorflow


class Solver(object):   
    def __init__(self, net ,data, rpn_loss, predict_loss):
        self.net = net
        self.data = data
        self.max_iter = cfg.MAX_ITER
        self.lr = cfg.LEARNING_RATE
        self.rpn_loss = rpn_loss
        self.predict_loss = predict_loss
        self.lr_change_ITER = cfg.lr_change_ITER
        self.summary_iter = cfg.SUMMARY_ITER
        self.save_iter = cfg.SAVE_ITER
        self._variables_to_fix = {}
        self.summary = os.path.join(cfg.Summary)
        if not os.path.exists(self.summary):
            os.mkdir(self.summary)
        self.ckpt = os.path.join(cfg.CKPT)
        if not os.path.exists(self.ckpt):
            os.mkdir(self.ckpt)
        self.ckpt_filename = os.path.join(self.ckpt, 'ckpt.model')
        

    def train_model(self):
        lr = tf.Variable(self.lr[0],trainable=False)
        self.optimizer = tf.train.MomentumOptimizer(lr, cfg.momentum)
        #self.optimizer = tf.train.GradientDescentOptimizer(lr)
        self.loss = self.rpn_loss.add_loss() + self.predict_loss.add_loss()     
        train_op = self.optimizer.minimize(self.loss)
        variables = tf.global_variables()
        reader = pywrap_tensorflow.NewCheckpointReader(self.net.weight_file_path)
        var_to_shape_map = reader.get_variable_to_shape_map()
        variables_to_restore = self.get_var_list(variables, var_to_shape_map)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver(var_list=variables_to_restore)
        merged = tf.summary.merge_all()
        with tf.Session() as sess:
            train_writer = tf.summary.FileWriter(self.summary, sess.graph)
            sess.run(init)
            saver.restore(sess, self.net.weight_file_path)
            self.fix_variables(sess, self.net.weight_file_path)
            saver = tf.train.Saver(variables, max_to_keep = 10)
            for step in range(self.max_iter+1):
                if step == self.lr_change_ITER:
                    lr = tf.assign(lr, self.lr[1])
                train_data = self.data.get()
                image_height = np.array(train_data['image'].shape[1])
                image_width = np.array(train_data['image'].shape[2])
                feed_dict = {self.net.image: train_data['image'], self.net.image_width: image_width,\
                             self.net.image_height: image_height, self.net.gt_boxes: train_data['box'],\
                             self.net.gt_cls: train_data['cls']}
                if step % self.summary_iter == 0:
                    total_loss, summary, learning_rate= sess.run([self.loss, merged, lr], feed_dict=feed_dict)
                    train_writer.add_summary(summary, step)
                    print('The', step, 'step train_total_loss is', total_loss)
                    print ('learning_rate is ', learning_rate)
                if step % self.save_iter == 0:
                    saver.save(sess, self.ckpt_filename, global_step = step)
                sess.run(train_op, feed_dict=feed_dict)
                    

    def get_var_list(self, global_variables, ckpt_variables):
        variables_to_restore = []
        for key in global_variables:
            print (key.name)
            if key.name == ('vgg_16/fc6/weights:0') or key.name == ('vgg_16/fc7/weights:0'):
                self._variables_to_fix[key.name] = key
                continue
            if key.name.split(':')[0] in ckpt_variables:
                variables_to_restore.append(key) 
        return variables_to_restore


    def fix_variables(self, sess, pretrained_model):
        print('Fix VGG16 layers..')
        with tf.variable_scope('Fix_VGG16') as scope:
            with tf.device("/cpu:0"):
                fc6_conv = tf.get_variable("fc6_conv", [7, 7, 512, 4096], trainable=False)
                fc7_conv = tf.get_variable("fc7_conv", [1, 1, 4096, 4096], trainable=False)
                restorer_fc = tf.train.Saver({'vgg_16' + "/fc6/weights": fc6_conv, 
                                              'vgg_16' + "/fc7/weights": fc7_conv})
                restorer_fc.restore(sess, pretrained_model)
                sess.run(tf.assign(self._variables_to_fix['vgg_16' + '/fc6/weights:0'], tf.reshape(fc6_conv, 
                                    self._variables_to_fix['vgg_16' + '/fc6/weights:0'].get_shape())))
                sess.run(tf.assign(self._variables_to_fix['vgg_16' + '/fc7/weights:0'], tf.reshape(fc7_conv, 
                                    self._variables_to_fix['vgg_16' + '/fc7/weights:0'].get_shape())))
    

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    net = network.Net()
    rpn_loss_obj = RPN_loss(net.rois_output['rois_bbx'], net.all_anchors, net.gt_boxes, \
                        net.rois_output['rois_cls'], net.labels, net.anchor_obj)
    predict_loss = Predict_loss(net._predictions["cls_score"], net._proposal_targets['labels'],\
                                net._predictions['bbox_pred'], net._proposal_targets['bbox_targets'],\
                                net._proposal_targets['bbox_inside_weights'], net._proposal_targets['bbox_outside_weights'])
    train_data = pascl.pascal_voc('train', fliped=False)
    solver = Solver(net, train_data, rpn_loss_obj, predict_loss)
    print ('start training')
    solver.train_model()