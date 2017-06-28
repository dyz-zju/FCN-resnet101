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
import skimage.io as io

MEAN_VALUE = np.array([103.939, 116.779, 123.68])   # BGR

#preprocess the input image and the annotation
def preprocess(img, anno):
    # img is (channels, height, width), values are 0-255bj
    for i in range(FLAGS.batch_size):
        img = img.astype(float)
        r, g, b = cv2.split(img[i])
        img[i] = cv2.merge([b, g, r])
        img[i] -= MEAN_VALUE
        #########################random flip#############################
        r = np.random.randint(2)
        if r == 0:
            img[i] = img[i]
            anno[i] = anno[i]
        else:
            img[i] = np.fliplr(img[i])
            anno[i] = np.fliplr(anno[i])
    return img, anno


def preprocess_testimg(filename):
    # prepare test image
    img = cv2.imread(filename)
    img = cv2.resize(img,(FLAGS.image_width,FLAGS.image_height))
    img = img.astype(float)
    r, g, b = cv2.split(img)
    img = cv2.merge([b, g, r])
    img -= MEAN_VALUE
    return img

#convert the annotation into onehot
def convert2onehot(a, classnum):
    onehot = np.zeros((a.shape[0], a.shape[1], a.shape[2], classnum))
    for batch in range(a.shape[0]):
        for width in range(a.shape[1]):
            for height in range(a.shape[2]):
                onehot[batch][width][height][
                    int(a[batch][width][height])] = 1.0
    return onehot

#####




#add color to the picture using the palette in VOC datasets
def palette(img):
    res_colors=[]
    for b in range(FLAGS.batch_size):
        res_color=np.zeros((FLAGS.image_width,FLAGS.image_height,3))
        for i in range(FLAGS.image_width):
            for j in range(FLAGS.image_height):
                res_color[i][j]=palette_label[img[b][i][j]]
        res_colors.append(res_color)
    return np.array(res_colors)


if __name__ == '__main__':
    #set the test false. Input will be different in train and test
    FLAGS.test=False
    #make sure tfrecords is exists.
    tfrecords_filename = './data/pascal_voc_segmentation.tfrecords'
    print("THE TFRECORDS IS EXISTED:", os.path.isfile(tfrecords_filename))
    #using a queue reader
    filename_queue = tf.train.string_input_producer(
        [tfrecords_filename], num_epochs=100)
    image, annotation = read_and_decode(filename_queue)

    #input
    input_image = tf.placeholder(
        tf.float32, [FLAGS.batch_size, FLAGS.image_width,FLAGS.image_height, 3], name="input")
    #annotation
    label_image = tf.placeholder(
        tf.float32, [FLAGS.batch_size, FLAGS.image_width,FLAGS.image_height, 21], name="label")
    #build resnet101
    net = network.Network(input_image)
    #build networks behind resnet
    res = net.build_networks()
    #define cost
    cost = net.buildloss(label_image)

    # writer
    tf.summary.scalar('cost', cost)

    train_op = tf.train.AdamOptimizer(
        learning_rate=1e-4).minimize(cost)
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    #save ckpt model
    saver = tf.train.Saver() 
    # test_img
    test_imgs = []
    test_imgs.append(preprocess_testimg(FLAGS.test_image1))
    test_imgs.append(preprocess_testimg(FLAGS.test_image2))

    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('/tmp/logs', sess.graph)
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        restore(sess)
        for i in range(FLAGS.max_iters):
            img, anno = sess.run([image, annotation])
            img, anno = preprocess(img, anno)
            # print(anno.shape)
            anno_1hot = convert2onehot(anno, 21)
            # print("anno shape", anno_1hot.shape)
            # print("test anno", anno_1hot[0][0][0])
            summary, loss, _ = sess.run([merged, cost, train_op], feed_dict={
                input_image: img, label_image: anno_1hot})

            
            if (i + 1) % 100 == 0:
                print("%d / %d iterations, the loss: %f" %
                      (i + 1, FLAGS.max_iters, loss))
                train_writer.add_summary(summary, i)

            if (i + 1) % 1000 == 0:
                saver.save(sess, FLAGS.train_dir+"/model_%d_iterations.ckpt"%(i+1))
                print("SUCCESSFULLY SAVED "+FLAGS.train_dir+"/model_%d_iterations.ckpt"%(i+1))
                test_res = sess.run(res, feed_dict={input_image: test_imgs})
                test_res = np.argmax(test_res, axis=3)
                test_res_colors=palette(test_res)

                # print(test_res_colors.shape)
                # print(test_res_colors)
                scipy.misc.imsave('./data/demo/test1_%d_iters.png' % (i + 1), test_res_colors[0])
                scipy.misc.imsave('./data/demo/test2_%d_iters.png' % (i + 1), test_res_colors[1])
