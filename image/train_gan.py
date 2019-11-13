from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import math
import numpy as np
import os
import scipy

from absl import flags
from absl import logging
import tensorflow as tf

import resnet

slim = tf.contrib.slim
tfgan = tf.contrib.gan

flags.DEFINE_integer('batch_size', 32, 'The number of images in each batch.')

flags.DEFINE_string('train_dir', '',
                    'Directory where to write event logs.')

flags.DEFINE_string('dataset_dir', None, 'Location of data.')

flags.DEFINE_integer('dataset_size', 32,
                     'The width and height of the images to be generated.')

flags.DEFINE_integer('max_number_of_steps', 100000,
                     'The maximum number of gradient steps.')

FLAGS = flags.FLAGS


def main(_):
  if not tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.MakeDirs(FLAGS.train_dir)

  # Force all input processing onto CPU in order to reserve the GPU for
  # the forward inference and back-propagation.
  with tf.name_scope('inputs'):
    with tf.device('/cpu:0'):
      images = glob.glob(os.path.join(FLAGS.dataset_dir, str(FLAGS.dataset_size), '*.png'))

      dataset = tf.data.Dataset.from_tensor_slices(images).map(lambda image: tf.image.decode_png(tf.read_file(image))).map(lambda image: (tf.to_float(image) - 128.0) / 128.0)
      dataset = dataset.repeat().batch(FLAGS.batch_size, drop_remainder=True).apply(tf.contrib.data.assert_element_shape(tf.TensorShape([FLAGS.batch_size, FLAGS.dataset_size, FLAGS.dataset_size, 3])))

      iterator = dataset.make_one_shot_iterator()
      images = iterator.get_next()

  # Define the GANModel tuple.
  generator_fn = lambda noise: tf.tanh(resnet.generator(noise, is_training=True, fused_batch_norm=True, size=FLAGS.dataset_size))
  discriminator_fn = lambda img, _: resnet.discriminator(img, is_training=True, fused_batch_norm=True, size=FLAGS.dataset_size)

  generator_inputs = tf.random_normal([FLAGS.batch_size, 64])
  gan_model = tfgan.gan_model(
      generator_fn,
      discriminator_fn,
      real_data=images,
      generator_inputs=generator_inputs)
  tfgan.eval.add_gan_model_image_summaries(gan_model)

  # Get the GANLoss tuple. Use the selected GAN loss functions.
  # (joelshor): Put this block in `with tf.name_scope('loss'):` when
  # cl/171610946 goes into the opensource release.
  gan_loss = tfgan.gan_loss(gan_model,
                            gradient_penalty_weight=1.0,
                            add_summaries=True)

  # Get the GANTrain ops using the custom optimizers and optional
  # discriminator weight clipping.
  with tf.name_scope('train'):
    gen_lr, dis_lr = _learning_rate()
    train_ops = tfgan.gan_train_ops(
        gan_model,
        gan_loss,
        generator_optimizer=tf.train.AdamOptimizer(gen_lr, 0.5),
        discriminator_optimizer=tf.train.AdamOptimizer(dis_lr, 0.5),
        summarize_gradients=True,
        colocate_gradients_with_ops=True,
        aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
    tf.summary.scalar('generator_lr', gen_lr)
    tf.summary.scalar('discriminator_lr', dis_lr)

  status_message = tf.string_join(
      ['Starting train step: ',
       tf.as_string(tf.train.get_or_create_global_step())],
      name='status_message')
  if FLAGS.max_number_of_steps == 0: return
  tfgan.gan_train(
      train_ops,
      hooks=(
          [tf.train.StopAtStepHook(num_steps=FLAGS.max_number_of_steps),
           tf.train.LoggingTensorHook([status_message], every_n_iter=10)]),
      logdir=FLAGS.train_dir)


def _learning_rate():
  generator_lr = tf.train.exponential_decay(
      learning_rate=0.0001,
      global_step=tf.train.get_or_create_global_step(),
      decay_steps=100000,
      decay_rate=0.9,
      staircase=True)
  discriminator_lr = 0.001
  return generator_lr, discriminator_lr


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  tf.app.run()

