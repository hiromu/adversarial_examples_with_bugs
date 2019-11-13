import numpy as np
import sys
import tensorflow as tf

from tensorflow.contrib.slim.nets import inception
slim = tf.contrib.slim

with tf.Graph().as_default() as graph:
    image = tf.placeholder(tf.float32, [1, 299, 299, 3])
    with slim.arg_scope(inception.inception_v3_arg_scope()):
        logits, _ = inception.inception_v3(image, num_classes=1001, is_training=False)

    init_fn = tf.contrib.slim.assign_from_checkpoint_fn(sys.argv[1], tf.contrib.slim.get_model_variables('InceptionV3'))
    with tf.Session(graph=graph) as sess:
        init_fn(sess)
        saver = tf.train.Saver(tf.global_variables())
        saver.save(sess, sys.argv[2])
