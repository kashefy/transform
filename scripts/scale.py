import os
import tempfile
import shutil
import numpy as np
from nose.tools import assert_equals, assert_true, assert_false,     assert_list_equal
from mock import patch
from nideep.datasets.mnist.mnist_tf import MNIST
import numpy as np
import transform.augmentation as aug


import tensorflow as tf

grph = tf.Graph()
with grph.as_default() as g:
    with tf.Session(graph=g) as sess:

        data = MNIST.read_data_sets('MNIST_data', one_hot=True)
        min_scale = 0.5
        max_scale = 2.5
        delta_scale = 0.3
        batch_sz = 10
        name = 'scale'
        batch_size_train = batch_sz
        new_dims = aug.scaled_dims(28, 28, min_scale, max_scale, delta_scale)
        print("new_dims", new_dims)

        x = tf.placeholder(tf.float32, [None, 784])
        augment_op, conds, bbox_top_left = aug.scale_translate_ops(x, new_dims,
            batch_sz, name, target_shape=[-1, 28, 28, 1])
        num_conds = len(conds)
        print("conds", conds)
        max_height, max_width = np.max(np.array(conds), axis=0)

        init_op = tf.global_variables_initializer()

        batch_xs, batch_ys = data.train.next_batch(batch_size_train)

        batch_xs_in = sess.run(augment_op, feed_dict={x: batch_xs})
        print(batch_xs_in.shape)

        import matplotlib.pyplot as plt
        f, a = plt.subplots(1, batch_size_train, figsize=(10, 2))
        for idx, v in enumerate(batch_xs):
            a[idx].imshow(v.reshape(28, 28))
        f, a = plt.subplots(1, len(batch_xs_in), figsize=(10, 2))
        for idx, v in enumerate(batch_xs_in):
            a[idx].imshow(v.reshape(max_height, max_width))
        plt.show()
