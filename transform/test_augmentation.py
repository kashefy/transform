'''
Created on Aug 11, 2017

@author: kashefy
'''
import os
import tempfile
import shutil
import numpy as np
from nose.tools import assert_equals, assert_true, assert_false, \
    assert_list_equal
from mock import patch
from nideep.datasets.mnist.mnist_tf import MNIST
import numpy as np
from transform.augmentation import scale_ops

class TestMNISTTF:
    @classmethod
    def setup_class(self):
        self.dir_tmp = tempfile.mkdtemp()

    @classmethod
    def teardown_class(self):
        shutil.rmtree(self.dir_tmp)

    def test_to_tf_record(self):
        import tensorflow as tf
        x = tf.placeholder(tf.float32, [None, 784])
        name_w = 'W'
        with tf.variable_scope("var_scope", reuse=None):
            W = tf.get_variable(name_w, shape=[784, 10],
                                initializer=tf.random_normal_initializer(stddev=0.1))
            b = tf.get_variable('b', shape=[10],
                                initializer=tf.constant_initializer(0.1))
        logits = tf.matmul(x, W) + b
        y = tf.nn.softmax(logits)
        y_ = tf.placeholder(tf.float32, [None, 10])
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits( \
                labels=y_, logits=logits))
        train_step = tf.train.GradientDescentOptimizer(0.9).minimize(cross_entropy)
        init_op = tf.global_variables_initializer()
        one_hot = True
        shuffle = True
        data = MNIST.read_data_sets('MNIST_data', one_hot=True)
        augment_op, conds = \
            scale_ops(x, min_scale, max_scale, delta_scale, batch_sz, name, target_shape=[-1, 28, 28, 1])
        num_conds = len(conds)

        batch_xs, batch_ys = self.data.train.next_batch(self.batch_size_train)

        batch_xs_in = sess.run(augment_op, feed_dict={self.x: batch_xs})
        _, _, _, _ = sess.run( \
            [self._acc_ops.metric, self._acc_ops.update,
             self._acc_orient_ops.metric, self._acc_orient_ops.update,
             #                             tf.argmax(self.model.p,1), tf.argmax(self.y_,1),
             ],
            feed_dict={self.x: batch_xs_in,
                       self.y_: batch_ys,
                       }
        )

        import matplotlib.pyplot as plt
        f, a = plt.subplots(3, 12, figsize=(10, 2))