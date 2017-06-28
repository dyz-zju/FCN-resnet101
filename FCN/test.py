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
from utils.timer import Timer
import numpy as np
import scipy.misc
import glob
import skimage.io as io


MEAN_VALUE = np.array([103.939, 116.779, 123.68])   # BGR



def preprocess_testimg(filename):
    # prepare test image
    img = cv2.imread(filename)
    img = img.astype(float)
    # convert image into BGR
    r, g, b = cv2.split(img)
    img = cv2.merge([b, g, r])
    img -= MEAN_VALUE

    return img




def palette(img):
    # add grayscale into color using the plalette
    width=img.shape[1]
    height=img.shape[2]
    res_colors=[]
    res_color=np.zeros((width,height,3))
    for i in range(width):
        for j in range(height):
            res_color[i][j]=palette_label[img[0][i][j]]
    res_colors.append(res_color)
    return np.array(res_colors)






if __name__ == '__main__':
    # build the network
    input_image = tf.placeholder(
        tf.float32, [1, None, None, 3], name="input")

    net = network.Network(input_image)

    res = net.build_networks()


    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    # fetch the images in the /test/demo
    im_names = glob.glob(os.path.join('./test/demo', '*.png')) + \
               glob.glob(os.path.join('./test/demo', '*.jpg'))

    # if delete this code, it will go wrong.I have no idea.
    saver = tf.train.Saver() 


    with tf.Session() as sess:
        sess.run(init_op)
        restore(sess)
        for im_name in im_names:
            print ('-----------------------------------------')
            # print ('Demo for %s'%(im_name))
            timer = Timer()
            timer.tic()
            # turn the image into [1,?,?,3]
            test_imgs = []
            test_imgs.append(preprocess_testimg(im_name))
            test_res = sess.run(res, feed_dict={input_image: test_imgs})
            timer.toc()
            test_res = np.argmax(test_res, axis=3)
            test_res_colors=palette(test_res)
            # save image
            scipy.misc.imsave('%s_seg.png' % (im_name[:-4]), test_res_colors[0])
            print("Saved image and cost %3fs"%(timer.total_time))

