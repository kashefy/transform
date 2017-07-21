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
import matplotlib.pyplot as plt
import tensorflow as tf
from ae_runner import AERunner
from mlp_runner import MLPRunner
from stacked_autoencoder_tf import StackedAutoencoder as SAE
from nideep.nets.mlp_tf import MLP
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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

def run_autoencoder(args):
    
    logger.info("Starting run %s" % args.run_name)
    # -90 (cw) to 90 deg (ccw) rotations in 15-deg increments
    rotations = np.deg2rad(np.linspace(-90, 90, 180/(12+1), endpoint=True)).tolist()    
    ae_runner = AERunner(args)
    n_input = ae_runner.data.train.images.shape[-1]
    sae_params = {
            'in_op': tf.placeholder("float", [None, n_input]),
            'prefix': 'sae',
            }
    ae_runner.model = SAE(sae_params)
    mlp_runner = MLPRunner(args)

    # Launch the graph
    with tf.Session() as sess:
        ae_runner.learn(sess)
        
        n_classes = mlp_runner.data.train.labels.shape[-1]
        classifier_params = {
            'n_nodes': [n_classes],
            'n_input': ae_runner.model.representation.get_shape()[-1].value,
            'prefix': 'mlp_',
            }
        net = MLP(classifier_params)
        net.x = ae_runner.model.representation
        net.build()
        mlp_runner.x = ae_runner.model.x
        mlp_runner.model = net
        mlp_runner.learn(sess)
        
        logger.debug('encoder-0 %s:' % sess.run(ae_runner.model.sae[0].w['encoder-0/w'][10,5:10]))
        #finetune(args, sess, sae)
    logger.info("Finished run %s" % args.run_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--learning_rate", type=float, default=0.01,
                        help="Set base learning rate")
    parser.add_argument("-e", "--epochs", dest="training_epochs", type=int, default=2, 
                        help="Set no. of epochs per layer")
    parser.add_argument("-b", "--batch_size", type=int, default=128,
                        help="Set mini-batch size")
    parser.add_argument("--display_step", type=int, default=1,
                        help="Set display step in no. of epochs")
    parser.add_argument("--n_disp", dest="examples_to_show", type=int, default=10,
                        help="Set no. of examples to show after training")
    parser.add_argument("--log_dir", dest="log_dir", type=str, default='/home/kashefy/models/ae/log_simple_stats',
                        help="Set parent log directory for all runs")
    parser.add_argument("--run_name", dest="run_name", type=str, default=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                        help="Set name for run")
    
    args = parser.parse_args()
    run_autoencoder(args)
    
    pass