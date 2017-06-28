from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

##########################
#                  restore
##########################
tf.app.flags.DEFINE_string(
    'train_dir', './output/',
    'Directory where checkpoints and event logs are written to.')

tf.app.flags.DEFINE_string(
    'pretrained_model', './data/pretrained_model/resnet_v1_101.ckpt',
    'Path to pretrained model')

tf.app.flags.DEFINE_string(
    'eval_dir', './eval/VOC2012_val/',
    'Directory where checkpoints and event logs are written to.')


tf.app.flags.DEFINE_integer(
    'blocks', 3,
    'numbers of used blocks')


tf.app.flags.DEFINE_string(
    'network', 'resnet101',
    'name of backbone network')


tf.app.flags.DEFINE_string(
    'dataset_name', 'Pascal VOC 2012',
    'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_tfrecords', 'pascal_voc_segmentation.tfrecords',
    'The name of the tfrecords.')

tf.app.flags.DEFINE_string(
    'dataset_dir', 'data/',
    'The directory where the dataset files are stored.')



tf.app.flags.DEFINE_integer(
    'max_iters', 30000,
    'max iterations')

#######################


tf.app.flags.DEFINE_string(
    'model_name', 'resnet101',
    'The name of the architecture to train.')


tf.app.flags.DEFINE_integer(
    'batch_size', 2,
    'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'image_width', 512,
    'The width of image.')

tf.app.flags.DEFINE_integer(
    'image_height', 512,
    'The height of image.')

tf.app.flags.DEFINE_string(
    'test_image1','./data/demo/test1.jpg',
    'the dir of test image1')

tf.app.flags.DEFINE_string(
    'test_image2','./data/demo/test2.jpg',
    'the dir of test image2')


tf.app.flags.DEFINE_boolean(
    'restore_previous_if_exists', True,
    'When restoring a checkpoint would ignore missing variables.')

tf.app.flags.DEFINE_boolean(
    'test', True,
    'When restoring a checkpoint would ignore missing variables.')



FLAGS = tf.app.flags.FLAGS
