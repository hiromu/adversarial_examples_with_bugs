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
from tensorflow.contrib.slim.nets import inception

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

flags.DEFINE_string('gan_model', '',
                    'The directory of the saved GAN model.')

flags.DEFINE_string('adv_image', '',
                    'The path of the original image.')

flags.DEFINE_string('adv_model', '',
                    'The directory of the saved target model.')

flags.DEFINE_integer('adv_target', 0,
                     'The label the adversarial example to be recognized.')

flags.DEFINE_integer('adv_filter', 200,
                     'The threshold to deal the perturbation as transparent.')

flags.DEFINE_float('adv_confidence', 0,
                   'The parameter specifing how strong the adversarial example should be.')

flags.DEFINE_float('adv_lambda', 5.0,
                   'The parameter to adjust the magnitude of the loss from the targeted model.')

FLAGS = flags.FLAGS


def main(_):
  image_orig = tf.image.decode_png(tf.read_file(FLAGS.adv_image))
  image_pad = (299 - FLAGS.dataset_size) / 2

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
  with tf.variable_scope('GAN'):
    generator_fn = lambda noise: tf.tanh(resnet.generator(noise, is_training=True, fused_batch_norm=True, size=FLAGS.dataset_size))
    discriminator_fn = lambda img, _: resnet.discriminator(img, is_training=True, fused_batch_norm=True, size=FLAGS.dataset_size)

  # Define variables for the image composition
  adv_theta = tf.random.uniform(shape=(FLAGS.batch_size, ), minval=0, maxval=math.pi * 2)
  adv_dx = tf.random.uniform(shape=(FLAGS.batch_size, ), minval=-image_pad, maxval=image_pad)
  adv_dy = tf.random.uniform(shape=(FLAGS.batch_size, ), minval=-image_pad, maxval=image_pad)

  generator_inputs = tf.random_normal([FLAGS.batch_size, 64])
  gan_model = tfgan.gan_model(
      generator_fn,
      discriminator_fn,
      real_data=images,
      generator_inputs=generator_inputs)
  tfgan.eval.add_gan_model_image_summaries(gan_model)

  def generator_loss(gan_model, *args, **kwargs):
    # Calculate the original loss from GAN
    wass_loss = tfgan.losses.wasserstein_generator_loss(gan_model, *args, **kwargs)

    # Synthesis input images
    adv_perbs = gan_model.generated_data * 128.0 + 128.0
    adv_perbs_mask = tf.cast((adv_perbs[:, :, :, 0] < FLAGS.adv_filter) | (adv_perbs[:, :, :, 1] < FLAGS.adv_filter) | (adv_perbs[:, :, :, 2] < FLAGS.adv_filter), tf.float32)
    adv_perbs_mask = tf.tile(tf.expand_dims(adv_perbs_mask, 3), [1, 1, 1, 3])
    adv_perbs_large = tf.pad(adv_perbs * adv_perbs_mask, [[0, 0]] + [[int(math.floor(image_pad)), int(math.ceil(image_pad))]] * 2 + [[0, 0]])

    adv_transform_rot = tf.Variable(np.zeros((FLAGS.batch_size, 3, 3)), dtype=tf.float32)
    adv_transform_rot = tf.scatter_nd_update(adv_transform_rot, [[x, 0, 0] for x in xrange(FLAGS.batch_size)], tf.math.cos(adv_theta))
    adv_transform_rot = tf.scatter_nd_update(adv_transform_rot, [[x, 0, 1] for x in xrange(FLAGS.batch_size)], -tf.math.sin(adv_theta))
    adv_transform_rot = tf.scatter_nd_update(adv_transform_rot, [[x, 0, 2] for x in xrange(FLAGS.batch_size)], (1 - tf.math.cos(adv_theta) + tf.math.sin(adv_theta)) * 299.0 / 2)
    adv_transform_rot = tf.scatter_nd_update(adv_transform_rot, [[x, 1, 0] for x in xrange(FLAGS.batch_size)], tf.math.sin(adv_theta))
    adv_transform_rot = tf.scatter_nd_update(adv_transform_rot, [[x, 1, 1] for x in xrange(FLAGS.batch_size)], tf.math.cos(adv_theta))
    adv_transform_rot = tf.scatter_nd_update(adv_transform_rot, [[x, 1, 2] for x in xrange(FLAGS.batch_size)], (1 - tf.math.cos(adv_theta) - tf.math.sin(adv_theta)) * 299.0 / 2)
    adv_transform_rot = tf.scatter_nd_update(adv_transform_rot, [[x, 2, 2] for x in xrange(FLAGS.batch_size)], tf.ones(shape=(FLAGS.batch_size, )))

    adv_transform_move = tf.Variable(np.zeros((FLAGS.batch_size, 3, 3)), dtype=tf.float32)
    adv_transform_move = tf.scatter_nd_update(adv_transform_move, [[x, 0, 0] for x in xrange(FLAGS.batch_size)], tf.ones(shape=(FLAGS.batch_size, )))
    adv_transform_move = tf.scatter_nd_update(adv_transform_move, [[x, 0, 2] for x in xrange(FLAGS.batch_size)], adv_dx)
    adv_transform_move = tf.scatter_nd_update(adv_transform_move, [[x, 1, 1] for x in xrange(FLAGS.batch_size)], tf.ones(shape=(FLAGS.batch_size, )))
    adv_transform_move = tf.scatter_nd_update(adv_transform_move, [[x, 1, 2] for x in xrange(FLAGS.batch_size)], adv_dy)
    adv_transform_move = tf.scatter_nd_update(adv_transform_move, [[x, 2, 2] for x in xrange(FLAGS.batch_size)], tf.ones(shape=(FLAGS.batch_size, )))

    adv_transform = tf.reshape(tf.matmul(adv_transform_rot, adv_transform_move), [-1, 9])[:, :8]
    adv_perbs_trans = tf.contrib.image.transform(adv_perbs_large, adv_transform)
    adv_perbs_trans_mask = tf.cast((adv_perbs_trans[:, :, :, 0] > 0) & (adv_perbs_trans[:, :, :, 1] > 0) & (adv_perbs_trans[:, :, :, 2] > 0), tf.float32)
    adv_perbs_trans_mask = tf.tile(tf.expand_dims(adv_perbs_trans_mask, 3), [1, 1, 1, 3])

    adv_images = tf.tile(tf.expand_dims(tf.cast(image_orig, tf.float32), 0), [FLAGS.batch_size, 1, 1, 1])
    adv_images = adv_perbs_trans * adv_perbs_trans_mask + adv_images * (1 - adv_perbs_trans_mask)
    adv_images_to_save = tf.identity(tf.map_fn(tf.image.encode_png, tf.cast(adv_images, tf.uint8), dtype=tf.string), name='adv_images_to_save')
    tf.summary.image('adv_images', tf.cast(adv_images, tf.uint8), max_outputs=FLAGS.batch_size)

    adv_inputs = adv_images / 255.0
    with tf.variable_scope('Inception'):
      with slim.arg_scope(inception.inception_v3_arg_scope()):
        adv_logits, _ = inception.inception_v3(adv_inputs, num_classes=1001, is_training=False)
    adv_logits_to_save = tf.identity(adv_logits, name='adv_logits_to_save')
    tf.summary.histogram('generator_adv_classes', tf.argmax(adv_logits, axis=1))

    # Add the adversarial loss
    with tf.name_scope('generator_loss'):
      # Came from: https://github.com/carlini/nn_robust_attacks/blob/master/l2_attack.py
      adv_targets = tf.one_hot(tf.constant([FLAGS.adv_target] * FLAGS.batch_size, dtype=tf.int32), 1001)
      adv_target_logit = tf.reduce_sum(adv_targets * adv_logits, 1)
      adv_others_logit = tf.reduce_max((1 - adv_targets) * adv_logits - (adv_targets * 10000), 1)

      adv_loss = tf.reduce_mean(tf.maximum(0.0, adv_others_logit - adv_target_logit + FLAGS.adv_confidence))
      tf.summary.scalar('generator_adv_loss', adv_loss)

      total_loss = wass_loss + FLAGS.adv_lambda * adv_loss
      tf.summary.scalar('generator_total_loss', total_loss)

    return total_loss

  # Get the GANLoss tuple. Use the selected GAN loss functions.
  # (joelshor): Put this block in `with tf.name_scope('loss'):` when
  # cl/171610946 goes into the opensource release.
  gan_loss = tfgan.gan_loss(gan_model,
                            generator_loss_fn=generator_loss,
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

  # Restore weights of the GAN and adversarial model
  adv_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Inception')
  adv_saver = tf.train.Saver(dict((var.name.replace('Inception/', '').replace(':0', ''), var) for var in adv_vars))
  gan_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Generator') + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Discriminator')
  gan_saver = tf.train.Saver(dict((var.name.replace(':0', ''), var) for var in gan_vars))
  class RestoreHook(tf.train.SessionRunHook):
    def after_create_session(self, session, coord):
      adv_saver.restore(session, FLAGS.adv_model)
      if FLAGS.gan_model:
        gan_saver.restore(session, FLAGS.gan_model)

  # Stops when certain numbers of adversarial examples are created
  class DumpHook(tf.train.SessionRunHook):
    def begin(self):
      self._count = 0
    def before_run(self, run_context):
      return tf.train.SessionRunArgs([tf.train.get_or_create_global_step(), run_context.session.graph.get_tensor_by_name('adv_images_to_save:0'), run_context.session.graph.get_tensor_by_name('adv_logits_to_save:0')])
    def after_run(self, run_context, run_values):
      global_step, images, logits = run_values.results
      images = images[np.argmax(logits, axis=1) == FLAGS.adv_target]
      if not os.path.exists(os.path.join(FLAGS.train_dir, 'images')):
        os.mkdir(os.path.join(FLAGS.train_dir, 'images'))
      for image in images:
        open(os.path.join(FLAGS.train_dir, 'images', '%03d_%06d.png' % (self._count, global_step)), 'wb').write(image)
        self._count += 1
      if self._count > 100:
        run_context.request_stop()

  status_message = tf.string_join(
      ['Starting train step: ',
       tf.as_string(tf.train.get_or_create_global_step())],
      name='status_message')
  if FLAGS.max_number_of_steps == 0: return
  tfgan.gan_train(
      train_ops,
      hooks=(
          [RestoreHook(), DumpHook(),
           tf.train.StopAtStepHook(num_steps=FLAGS.max_number_of_steps),
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

