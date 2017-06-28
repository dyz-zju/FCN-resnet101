from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import sys
path = os.path.abspath(os.path.join('..'))
sys.path.append(path)
import time
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from time import gmtime, strftime
from config.config_v1 import FLAGS
from tensorflow.python import pywrap_tensorflow

resnet_scope = 'resnet_v1_101'

def get_variables_to_restore(variables, var_keep_dic):
    variables_to_restore = []

    for v in variables:
        if v.name.split(':')[0] in var_keep_dic:
            print('Varibles restored: %s' % v.name)
            variables_to_restore.append(v)
    return variables_to_restore


def get_variables_in_checkpoint_file(file_name):
    try:
        reader = pywrap_tensorflow.NewCheckpointReader(file_name)
        var_to_shape_map = reader.get_variable_to_shape_map()
        return var_to_shape_map
    except Exception as e:  # pylint: disable=broad-except
        print(str(e))
        if "corrupted compressed block contents" in str(e):
            print("It's likely that your checkpoint file has been compressed "
                  "with SNAPPY.")


def restore(sess):
    #if there is no trained model , restore the resnet101 and train from the begining, otherwise, train from the previous model. 
    if FLAGS.restore_previous_if_exists:
        try:
            checkpoint_path = tf.train.latest_checkpoint(FLAGS.train_dir)
            restorer = tf.train.import_meta_graph(checkpoint_path+'.meta')
            restorer.restore(sess, checkpoint_path)
            print('\nRESTORED PREVIOUS MODEL %s FROM %s'
                  % (checkpoint_path, FLAGS.train_dir))
            time.sleep(2)
            return
        except:
            print("\nFAILED TO RESTORE IN %s %s\n" %
                  (FLAGS.train_dir, checkpoint_path))
            time.sleep(2)


    if FLAGS.pretrained_model:
        print("\nTHE CKPT IS EXISTED:", os.path.isfile(FLAGS.pretrained_model))
        try:
            variables = tf.global_variables()
            var_keep_dic = get_variables_in_checkpoint_file(FLAGS.pretrained_model)
            # Get the variables to restore, ignorizing the variables to fix
            variables_to_restore = get_variables_to_restore(variables, var_keep_dic)
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, FLAGS.pretrained_model)
            print("\nSUCCESSFULLY RESTORED: %s\n" % (FLAGS.pretrained_model))
        except:
            print("\nFAILED TO RESTORE  %s\n" % (FLAGS.pretrained_model))
            print("PLEASE CHECK YOUR FILES\n")
            raise
