from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tensorflow.examples.tutorials.mnist import input_data

import numpy as np
import tensorflow as tf
from config.config_v1 import FLAGS

#layers

class l_avgpool:

    def __init__(self, ksize, strides, padding, name):
        self.kind = "avgpool"
        self.ksize = ksize
        self.strides = strides
        self.padding = padding
        self.name = name


class l_maxpool:

    def __init__(self, ksize, strides, padding, name):
        self.kind = "maxpool"
        self.ksize = ksize
        self.strides = strides
        self.padding = padding
        self.name = name

# filter=[filter_height, filter_width, in_channels, out_channels]


class l_conv2d:

    def __init__(self, filter, strides, padding, name):
        self.kind = "conv2d"
        self.filter = tf.Variable(tf.truncated_normal(
            filter, dtype=tf.float32, stddev=1e-1, name='weights'))
        self.strides = strides
        self.padding = padding
        self.name = name


class l_deconv2d:

    def __init__(self, filter, strides, padding, name ,i):
        self.kind = "deconv2d"
        self.filter = tf.Variable(tf.truncated_normal(
            filter, dtype=tf.float32, stddev=1e-1, name='weights'))
        self.strides = strides
        self.padding = padding
        self.name = name
        self.i=i


class l_lrn:

    def __init__(self, depth_radius, bias, alpha, beta, name):
        self.kind = "lrn"
        self.depth_radius = depth_radius
        self.bias = bias
        self.alpha = alpha
        self.beta = beta
        self.name = name


class l_relu:

    def __init__(self, name):
        self.kind = "relu"
        self.name = name


class l_fc:

    def __init__(self, innode, outnode, name):
        self.kind = "fc"
        self.innode = innode
        self.outnode = outnode
        self.weight = tf.Variable(tf.truncated_normal(
            [self.innode, self.outnode], dtype=tf.float32, stddev=1e-1))
        self.bias = tf.Variable(tf.truncated_normal(
            [self.outnode], dtype=tf.float32, stddev=1e-1))
        self.name = name


class l_softmax:

    def __init__(self, name):
        self.kind = "softmax"
        self.name = name


class l_sigmoid:

    def __init__(self, name):
        self.kind = "sigmoid"
        self.name = name


class l_resnet_block:

    def __init__(self, innode_channel, outnode_channel, res_name, name, isChangeSize=False):
        self.kind = "resnet_block"
        self.innode_channel = innode_channel
        self.outnode_channel = outnode_channel
        self.res_name = res_name
        self.name = name
        self.isChangeSize = isChangeSize


class l_dropout:

    def __init__(self, keep_prob, name):
        self.kind = "dropout"
        self.keep_prob = keep_prob
        self.name = name


class l_batch_norm:

    def __init__(self, shape, name):
        self.kind = "batch_norm"
        self.gamma = tf.Variable(np.ones(shape, dtype='float32'), name='gamma')
        self.beta = tf.Variable(np.zeros(shape, dtype='float32'), name='beta')
        self.epsilon = np.float32(1e-5)
        self.name = name


def load_mnist():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    mnist_X = np.r_[mnist.train.images, mnist.test.images]
    mnist_y = np.r_[mnist.train.labels, mnist.test.labels]
    return train_test_split(mnist_X, mnist_y, test_size=0.2, random_state=42)



#inference

class Layers:

    def __init__(self, layers,net,images):
        self.layers = layers
        self.results = {}
        self.net=net
	self.images=images

    def inference(self, images):
        res = images
        for layer in self.layers:
            if layer.kind == 'maxpool':
                res = tf.nn.max_pool(
                    res, ksize=layer.ksize, strides=layer.strides, padding=layer.padding, name=layer.name)


            elif layer.kind == 'avgpool':
                res = tf.nn.avg_pool(
                    res, ksize=layer.ksize, strides=layer.strides, padding=layer.padding, name=layer.name)


            elif layer.kind == 'conv2d':
                bias = tf.Variable(
                    np.zeros((layer.filter.get_shape()[3]), dtype='float32'))
                res = tf.nn.bias_add(tf.nn.conv2d(
                    res, filter=layer.filter, strides=layer.strides, padding=layer.padding, name=layer.name), bias)


            elif layer.kind == 'deconv2d':
                shape = tf.shape(self.net)
                print("shape",shape)
                if FLAGS.test==True:
		    deconv_shape = tf.stack([1, shape[1]*layer.i, shape[2]*layer.i, layer.filter.get_shape()[2]])
                else:
                    deconv_shape = tf.stack([FLAGS.batch_size, shape[1]*layer.i, shape[2]*layer.i, layer.filter.get_shape()[2]])                    
                bias = tf.Variable(
                    np.zeros((layer.filter.get_shape()[2]), dtype='float32'))
                res = tf.nn.bias_add(tf.nn.conv2d_transpose(
                    res, filter=layer.filter, output_shape=deconv_shape, strides=layer.strides, padding=layer.padding, name=layer.name), bias)


            elif layer.kind == 'lrn':
                res = tf.nn.lrn(res, depth_radius=layer.depth_radius, bias=layer.bias,
                                alpha=layer.alpha, beta=layer.beta, name=layer.name)


            elif layer.kind == 'relu':
                res = tf.nn.relu(res, layer.name)


            elif layer.kind == 'fc':
                res = tf.reshape(
                    res, [-1, np.prod(res.get_shape().as_list()[1:])])
                res = tf.nn.bias_add(
                    tf.matmul(res, layer.weight), layer.bias)


            elif layer.kind == 'softmax':
                res = tf.nn.softmax(res, name=layer.name)


            elif layer.kind == 'sigmoid':
                res = tf.nn.sigmoid(res, name=layer.name)


            elif layer.kind == 'dropout':
                res = tf.nn.dropout(
                    res, keep_prob=layer.keep_prob, name=layer.name)


            elif layer.kind == 'batch_norm':
                if len(res.get_shape()) == 2:
                    mean, var = tf.nn.moments(res, axes=0, keep_dims=True)
                    std = tf.sqrt(var + layer.epsilon)
                elif len(res.get_shape()) == 4:
                    mean, var = tf.nn.moments(
                        res, axes=(0, 1, 2), keep_dims=True)
                    std = tf.sqrt(var + layer.epsilon)
                normalized_res = (res - mean) / std
                res = layer.gamma * normalized_res + layer.beta
            self.results[layer.name] = res
            print(res.shape)
        return res
