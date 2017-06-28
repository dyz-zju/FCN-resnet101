from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import sys
from nets import network
from config.config_v1 import FLAGS
from data.fetch_voc_data import read_and_decode
from utils.restore import restore
from utils.label_class_palette_utils import palette_label
import cv2
import numpy as np
import scipy.misc
import glob
import skimage.io as io


MEAN_VALUE = np.array([103.939, 116.779, 123.68])   # BGR


classes = {'aeroplane': 1,  'bicycle': 2,  'bird': 3,  'boat': 4,
           'bottle': 5,  'bus': 6,  'car': 7,  'cat': 8,
           'chair': 9,  'cow': 10, 'diningtable': 11, 'dog': 12,
           'horse': 13, 'motorbike': 14, 'person': 15, 'potted-plant': 16,
           'sheep': 17, 'sofa': 18, 'train': 19, 'tv/monitor': 20}


classes_reverse = {v: k for k, v in classes.items()}



def preprocess_testimg(filename):
    # prepare test image
    img = cv2.imread(filename)
    img = img.astype(float)
    # convert image into BGR
    r, g, b = cv2.split(img)
    img = cv2.merge([b, g, r])
    img -= MEAN_VALUE

    return img


def calculate_pads(width_pads,height_pads):
    if width_pads%2==0:
        width_pads_left=width_pads//2
        width_pads_right=width_pads//2
    else:
        width_pads_left=width_pads//2+1
        width_pads_right=width_pads//2
    if height_pads%2==0:
        height_pads_top=height_pads//2
        height_pads_bottom=height_pads//2
    else:
        height_pads_top=height_pads//2+1
        height_pads_bottom=height_pads//2      
    return [width_pads_left,width_pads_right,height_pads_top,height_pads_bottom]


f=open(FLAGS.eval_dir+"test.txt","r")
im_names=f.readlines()
f.close()

score_matrix=np.zeros((21,21))

if __name__ == '__main__':
    # build the network
    input_image = tf.placeholder(
        tf.float32, [1, None, None, 3], name="input")

    net = network.Network(input_image)

    res = net.build_networks()


    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())


    # if delete this code, it will go wrong.I have no idea.
    saver = tf.train.Saver() 

    with tf.Session() as sess:
        sess.run(init_op)
        restore(sess)
        for im_name in im_names:
            anno=cv2.imread(FLAGS.eval_dir+"Segmentation/"+im_name[:-1]+".png",cv2.IMREAD_GRAYSCALE)
            print ('-----------------------------------------')
            print("Evaluate pic name:%s"%(im_name[:-1]+".png"))
            test_imgs = []
            test_imgs.append(preprocess_testimg(FLAGS.eval_dir+"JPEGImages/"+im_name[:-1]+".jpg"))
            test_res = sess.run(res, feed_dict={input_image: test_imgs})
            test_res = np.argmax(test_res, axis=3)
            res_pic=test_res[0]
            #make the picture same size as original
            height_pads=res_pic.shape[0]-anno.shape[0]
            width_pads=res_pic.shape[1]-anno.shape[1]
            width_pads_left,width_pads_right,height_pads_top,height_pads_bottom=calculate_pads(width_pads,height_pads)
            for i in range(width_pads_left):
                    res_pic=np.delete(res_pic,0,axis=1)
            for i in range(width_pads_right):
                    res_pic=np.delete(res_pic,-1,axis=1)
            for i in range(height_pads_top):
                    res_pic=np.delete(res_pic,0,axis=0)
            for i in range(height_pads_bottom):
                    res_pic=np.delete(res_pic,-1,axis=0)
            # save image
            scipy.misc.imsave(FLAGS.eval_dir+'output/%s.png' % (im_name[:-1]), res_pic)
            for i in range(anno.shape[0]):
                for j in range(anno.shape[1]):
                    res_index=res_pic[i][j]
                    anno_index=anno[i][j]
                    score_matrix[res_index][anno_index]+=1
    accuracys=0
    #print(score_matrix)
    print("Evaluating ...")
    for i in range(1,21):
        true_positive=score_matrix[i][i]
        false_positive=np.sum(score_matrix,axis=1)[i]-score_matrix[i][i]
        false_negetive=np.sum(score_matrix,axis=0)[i]-score_matrix[i][i]
        accuracy=true_positive/(true_positive+false_positive+false_negetive)
        print("%s IoU accuracy : %f"%(classes_reverse[i],accuracy))
        accuracys=accuracys+accuracy
    print("Mean IoU accuracy= %f"%(accuracys/20))