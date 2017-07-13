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

import argparse
import os
import logging
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import matplotlib.pyplot as plt
from stacked_autoencoder_tf import StackedAutoencoder as SAE

logging.basicConfig(level=logging.DEBUG)

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
run_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def setup_optimizer_op(cost, learning_rate, var_list=None):
#    opt = tf.train.RMSPropOptimizer(learning_rate)
    opt = tf.train.GradientDescentOptimizer(learning_rate)
    op = opt.minimize(cost, var_list=var_list)
    return op

def train_layerwise(args, sess, sae):

    summary_writer = tf.summary.FileWriter(os.path.join(args.log_dir, run_dir),
                                           sess.graph)
    itr_exp = 0
    for dim in [256]:
        itr_layer = 0
        sae.stack(dim)
        for itrtemp in xrange(1):
            cost = sae.cost()
            
            vars_new = sae.vars_new()
#            logging.debug('new vars for optimizer %s' % [v.name for v in vars_new])
            optimizer = setup_optimizer_op(cost, args.learning_rate, var_list=vars_new)
            vars_new = sae.vars_new()
            
            # Initializing the variables
            #init_op = tf.global_variables_initializer()
            if itrtemp == 0:
                init_op = tf.variables_initializer(vars_new)
            
            
                logging.debug('initializing %s' % [v.name for v in vars_new])
                sess.run(init_op)
            total_batch = int(mnist.train.num_examples/args.batch_size)
            
            # Training cycle
    #        
            print('encoder-1', sess.run(sae.sae[0].w['encoder-1/w'][10,5:10]))
    #        if dim == 128:
    #            print('encoder_2', sess.run(sae.sae[1].w['encoder_2/w'][10,5:10]))
            
            for value in [cost]:
                print("log scalar", value.op.name)
                tf.summary.scalar(value.op.name, value)
            
            summaries = tf.summary.merge_all()
                
            for epoch in xrange(args.training_epochs):
                # Loop over all batches
                for itr_epoch in xrange(total_batch):
                    batch_xs, _ = mnist.train.next_batch(args.batch_size)
                    
                    
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
                    _, c, sess_summary = sess.run([optimizer, cost, summaries],
                                                  feed_dict={sae.x: batch_xs})
                    
                    
                    summary_writer.add_summary(sess_summary, itr_exp)
                    
                    itr_exp += 1
                    itr_layer += 1
                    
                # Display logs per epoch step
                if epoch % args.display_step == 0:
                    logging.info("Epoch: %04d, cost=%.9f" % (epoch+1, c))
                
            print("Optimization Finished!")
            print('encoder-1',sess.run(sae.sae[0].w['encoder-1/w'][10,5:10]))
#        if dim == 128:
#            print('encoder_2',sess.run(sae.sae[1].w['encoder_2/w'][10,5:10]))
        encode_decode = sess.run(
            sae.y_pred, feed_dict={sae.x: mnist.test.images[:args.examples_to_show]})
        # Compare original images with their reconstructions
    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in xrange(args.examples_to_show):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)), clim=(0.0, 1.0))
        a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)), clim=(0.0, 1.0))
    f.show()
        
def finetune(args, sess, sae):
    
    logging.info("Finetuning!")
#    print('encoder_1', sess.run(sae.sae[0].w['encoder_1/w'][10,5:10]))
#    print('encoder_2',sess.run(sae.sae[1].w['encoder_2/w'][10,5:10]))
    vars_old = [var.name for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)]
    with tf.name_scope('finetune'):
        cost = sae.sae[0].cost_cross_entropy(sae.y_pred, name='cost_finetune')
        optimizer = setup_optimizer_op(cost, args.learning_rate)
    vars_new = [var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if var.name not in vars_old]
    
    logging.debug("Initializing %s" % [var.name for var in vars_new])
    init_op = tf.variables_initializer(vars_new)
    sess.run(init_op)
    summary_writer = tf.summary.FileWriter(os.path.join(args.log_dir, run_dir), sess.graph)
    
    total_batch = int(mnist.train.num_examples/args.batch_size)
    # Training cycle
    encode_decode = sess.run(
        sae.y_pred, feed_dict={sae.x: mnist.test.images[:args.examples_to_show]})
        # Compare original images with their reconstructions
    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in xrange(args.examples_to_show):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)), clim=(0.0, 1.0))
        a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)), clim=(0.0, 1.0))
    f.show()
#    sess = tf_debug.LocalCLIDebugWrapperSession(sess)
#    for epoch in xrange(args.training_epochs):
    for epoch in xrange(1):
        # Loop over all batches
        for itr in xrange(total_batch):
            batch_xs, _ = mnist.train.next_batch(args.batch_size)
            _, c, rec = sess.run([optimizer, cost, sae.y_pred], feed_dict={sae.x: batch_xs})
            logging.info("%d/%d, cost=%.9f" % (itr, total_batch, c))
#            if c < 0.08:
            if itr % 10 == 0:
                encode_decode = sess.run(
                    sae.y_pred, feed_dict={sae.x: mnist.test.images[:args.examples_to_show]})
                f, a = plt.subplots(4, 10, figsize=(10, 2))
                for i in xrange(args.examples_to_show):
                    a[0][i].imshow(np.reshape(batch_xs[i], (28, 28)), clim=(0.0, 1.0))
                    a[1][i].imshow(np.reshape(rec[i], (28, 28)), clim=(0.0, 1.0))
                    a[2][i].imshow(np.reshape(mnist.test.images[i], (28, 28)), clim=(0.0, 1.0))
                    a[3][i].imshow(np.reshape(encode_decode[i], (28, 28)), clim=(0.0, 1.0))
    #            f.show()
                f.savefig("/home/kashefy/models/ae/out/%04d_%04d_%.9f.png" % (itr, total_batch, c))
                plt.clf()
        # Display logs per epoch step
        if epoch % args.display_step == 0:
            logging.info("Epoch: %04d, cost=%.9f" % (epoch+1, c))
        if epoch == 0:
            encode_decode = sess.run(
                sae.y_pred, feed_dict={sae.x: mnist.test.images[:args.examples_to_show]})
            f, a = plt.subplots(2, 10, figsize=(10, 2))
            for i in xrange(args.examples_to_show):
                a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)), clim=(0.0, 1.0))
                a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)), clim=(0.0, 1.0))
            f.show()

    print("Finetuning Finished!")
#    
#    print('encoder_1', sess.run(sae.sae[0].w['encoder_1/w'])[10,5:10])
#    print('encoder_2',sess.run(sae.sae[1].w['encoder_2/w'][10,5:10]))
    # Applying encode and decode over test set
    encode_decode = sess.run(
        sae.y_pred, feed_dict={sae.x: mnist.test.images[:args.examples_to_show]})
    # Compare original images with their reconstructions
    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in xrange(args.examples_to_show):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)), clim=(0.0, 1.0))
        a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)), clim=(0.0, 1.0))
    f.show()
    
def classification(args, sess, sae):
    summary_writer = tf.summary.FileWriter(os.path.join(args.log_dir, run_dir),
                                           sess.graph)
    
    n_classes = mnist.test.labels.shape[-1]
    
    y_ = tf.placeholder("float", [None, n_classes])
    itr_exp = 0
    depth = 1
    var_scope = 'dense-%d' %  depth
    name_scope = var_scope + '/'
    with tf.variable_scope(var_scope):
        fc_name_w = 'fc-%d/w' % depth
        w = {
            fc_name_w: tf.get_variable(fc_name_w,
                                       [sae.sae[-1].n_hidden_1, n_classes],
                                       initializer=tf.random_normal_initializer(),
                                       )
        }
        b = {}
        for key, value in w.iteritems():
            key_b = key.replace('/w', '/b').replace('_w', '_b').replace('-w', '-b')
            b[key_b] = tf.get_variable(key_b,
                                       [int(value.get_shape()[-1])],
                                       initializer=tf.constant_initializer(0.)
                                       )
    with tf.name_scope(name_scope + 'fc'):
        fc_name_b = 'fc-%d/b' % depth
        fc_op = tf.add(tf.matmul(sae.sae[-1].representation(), w[fc_name_w]),
                       b[fc_name_b],
                       name='fc-%d' % depth)
    y = fc_op
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
        
    vars_new = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                 scope=name_scope)
#            logging.debug('new vars for optimizer %s' % [v.name for v in vars_new])
    optimizer = setup_optimizer_op(cost, args.learning_rate, var_list=vars_new)
    vars_new = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                 scope=name_scope)
            
    # Initializing the variables
    init_op = tf.variables_initializer(vars_new)
    logging.debug('initializing %s' % [v.name for v in vars_new])
    sess.run(init_op)
    total_batch = int(mnist.train.num_examples/args.batch_size)
            
    # Training cycle
#        
    print('encoder-1', sess.run(sae.sae[0].w['encoder-1/w'][10,5:10]))
#        if dim == 128:
#            print('encoder_2', sess.run(sae.sae[1].w['encoder_2/w'][10,5:10]))
    
    for value in [cost]:
        print("log scalar", value.op.name)
        tf.summary.scalar(value.op.name, value)
    
    summaries = tf.summary.merge_all()
        
    for epoch in xrange(args.training_epochs):
        # Loop over all batches
        for itr_epoch in xrange(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(args.batch_size)
            _, c, sess_summary = sess.run([optimizer, cost, summaries],
                                          feed_dict={sae.x: batch_xs,
                                                     y_: batch_ys})
            
            summary_writer.add_summary(sess_summary, itr_exp)
            
            itr_exp += 1
            
        # Display logs per epoch step
        if epoch % args.display_step == 0:
            logging.info("Epoch: %04d, cost=%.9f" % (epoch+1, c))
    print("Classification Optimization Finished!")
    print('encoder-1',sess.run(sae.sae[0].w['encoder-1/w'][10,5:10]))
#        if dim == 128:
#            print('encoder_2',sess.run(sae.sae[1].w['encoder_2/w'][10,5:10]))

def run_autoencoder(args):
    
    # -90 (cw) to 90 deg (ccw) rotations in 15-deg increments
    rotations = np.deg2rad(np.linspace(-90, 90, 180/(12+1), endpoint=True)).tolist()
    
    n_input = mnist.test.images.shape[-1]
    
    X = tf.placeholder("float", [None, n_input])
    
    sae_params = {
        'in_op': X,
        }
    sae = SAE(sae_params)
    

    # Launch the graph
    with tf.Session() as sess:
        train_layerwise(args, sess, sae)
        classification(args, sess, sae)
        finetune_classification(args, sess, sae)
        #finetune(args, sess, sae)
        
#    plt.draw()
#    plt.waitforbuttonpress()
        
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--learning_rate", type=float, default=0.01,
                        help="Set base learning rate")
    parser.add_argument("-e", "--epochs", dest="training_epochs", type=int, default=60, 
                        help="Set no. of epochs per layer")
    parser.add_argument("-b", "--batch_size", type=int, default=128,
                        help="Set mini-batch size")
    parser.add_argument("--display_step", type=int, default=1,
                        help="Set display step in no. of epochs")
    parser.add_argument("--n_disp", dest="examples_to_show", type=int, default=10,
                        help="Set no. of examples to show after training")
    parser.add_argument("--log_dir", dest="log_dir", type=str, default='/home/kashefy/models/ae/log_simple_stats',
                        help="Set parent log directory for all runs")
    
    args = parser.parse_args()
    run_autoencoder(args)
    
    pass