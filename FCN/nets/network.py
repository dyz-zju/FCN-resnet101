import tensorflow as tf
import resnet_v1
import resnet_utils
import layer_utils
import sys
import os
path = os.path.abspath(os.path.join('..'))
sys.path.append(path)
import tensorflow.contrib.slim as slim
from config.config_v1 import FLAGS
from tensorflow.python.framework import ops
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.python.ops import nn_ops
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import layers


class Network(object):
    ##load the resnet101
    def __init__(self, images):
        self.layer = {}
        self.images = images

        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            self.nets, _ = resnet_v1.resnet_v1_101(self.images,
                                                1000,
                                                is_training=False,
                                                spatial_squeeze=False,
                                                global_pool=False,
                                                output_stride=16)
            print(len(self.nets))
            for index in range(len(self.nets)):
                print("resnet_bolck_%d"%(index+1))
                print(self.nets[index].get_shape())
        self.layer['block1']=self.nets[0]
        self.layer['block2']=self.nets[1]
        self.layer['block3']=self.nets[2]
        self.layer['block4'] = self.nets[3]


    #add layers 
    def build_networks(self):
        layers = [
                  layer_utils.l_deconv2d([3, 3, 512, 1000],[1,2,2,1], "SAME", "deconv1",i=1),
                  layer_utils.l_relu("relu2"),
                  layer_utils.l_deconv2d([3, 3, 21, 512],[1,2,2,1], "SAME", "deconv2",i=2),
                  layer_utils.l_relu("relu3"),
                  layer_utils.l_deconv2d([3, 3, 21, 21],[1,2,2,1], "SAME", "deconv3",i=4),
                  layer_utils.l_deconv2d([3, 3, 21, 21],[1,2,2,1], "SAME", "deconv4",i=8)
                  ]
        net = layer_utils.Layers(layers,self.nets[0],self.images)
        self.res = net.inference(self.layer['block4'])
        return self.res

    #define loss
    def buildloss(self,labels):
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.res, labels=labels))
        return self.cost