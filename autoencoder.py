# -*- coding: utf-8 -*-

""" Auto Encoder Example.
Using an auto encoder on MNIST handwritten digits.
References:
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.
Links:
    [MNIST Dataset] http://yann.lecun.com/exdb/mnist/
"""
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from stacked_autoencoder_tf import StackedAutoencoder as SAE

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# Parameters
learning_rate = 0.01
training_epochs = 10
batch_size = 256
display_step = 1
examples_to_show = 10

# -90 (cw) to 90 deg (ccw) rotations in 15-deg increments
rotations = np.deg2rad(np.linspace(-90, 90, 180/(12+1), endpoint=True)).tolist()

n_input = mnist.test.images.shape[-1]

X = tf.placeholder("float", [None, n_input])

sae = []

sae_params = {
    'dims': [512],
    'in_op': X,
    }
sae = SAE(sae_params)
sae.stack()
cost = sae.cost()

optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)


vars_new = sae.vars_new()
print('prev', [v.name for v in vars_new])

# Initializing the variables
#init_op = tf.global_variables_initializer()
init_op = tf.variables_initializer(vars_new)

summaries = tf.summary.merge_all()

# Launch the graph
with tf.Session() as sess:
    sess.run(init_op)
    summary_writer = tf.summary.FileWriter('/home/kashefy/models/ae/log_simple_stats', sess.graph)
    total_batch = int(mnist.train.num_examples/batch_size)
    # Training cycle
    for epoch in range(training_epochs):
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, _ = mnist.train.next_batch(batch_size)
            
            
#            batch_xs_as_img = tf.reshape(batch_xs, [-1, 28, 28, 1])
#            rots_cur = np.random.choice(rotations, batch_size)
#            batch_xs_as_img_rot = tf.contrib.image.rotate(batch_xs_as_img, rots_cur)
#            # Run optimization op (backprop) and cost op (to get loss value)
#            
#            batch_xs_as_img, batch_xs_as_img_rot = \
#                sess.run([batch_xs_as_img, batch_xs_as_img_rot],
#                         feed_dict={ae1.x: batch_xs})
#            f, a = plt.subplots(2, 10, figsize=(10, 2))
#            for i in xrange(examples_to_show):
#                print (batch_xs_as_img[i].shape, np.rad2deg(rots_cur)[i])
#                a[0][i].imshow(np.squeeze(batch_xs_as_img[i]))
#                a[1][i].imshow(np.squeeze(batch_xs_as_img_rot[i]))
#            f.show()
#            plt.draw()
#            plt.waitforbuttonpress()
#            
            
            
            _, c = sess.run([optimizer, cost], feed_dict={sae.x(): batch_xs})
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.9f}".format(c))

    print("Optimization Finished!")

    # Applying encode and decode over test set
    encode_decode = sess.run(
        sae.y_pred(), feed_dict={sae.x(): mnist.test.images[:examples_to_show]})
    # Compare original images with their reconstructions
    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in xrange(examples_to_show):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)), clim=(0.0, 0.1))
        a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)), clim=(0.0, 0.1))
    f.show()
    plt.draw()
    plt.waitforbuttonpress()