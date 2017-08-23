import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

def create(x, layer_sizes):

    # Build the encoding layers
    next_layer_input = x

    encoding_matrices = []
    for dim in layer_sizes:
        input_dim = int(next_layer_input.get_shape()[1])

        # Initialize W using random values in interval [-1/sqrt(n) , 1/sqrt(n)]
        W = tf.Variable(xavier_init([input_dim, dim]))
                        
        # Initialize b to zero
        b = tf.Variable(tf.zeros([dim]))

        # We are going to use tied-weights so store the W matrix for later reference.
        encoding_matrices.append(W)

        output = tf.nn.sigmoid(tf.matmul(next_layer_input, W) + b)

        # the input into the next layer is the output of this layer
        next_layer_input = output
        print input_dim, W.get_shape(), b.get_shape(), output.get_shape()

    # The fully encoded x value is now stored in the next_layer_input
    encoded_x = next_layer_input

    # build the reconstruction layers by reversing the reductions
    layer_sizes.reverse()
    encoding_matrices.reverse()

    for i, dim in enumerate(layer_sizes[1:] + [int(x.get_shape()[1])]) :
        # we are using tied weights, so just lookup the encoding matrix for this step and transpose it
        W = tf.transpose(encoding_matrices[i])
        b = tf.Variable(tf.zeros([dim]))
        output = tf.matmul(next_layer_input, W) + b
        output_transfer = tf.nn.sigmoid(output)
        next_layer_input = output_transfer
        print dim, W.get_shape(), b.get_shape(), output.get_shape()

    # the fully encoded and reconstructed value of x is here:
    reconstructed_x = next_layer_input
    
    loss = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(reconstructed_x, x), 2.0))
#    tf.reduce_sum(\
#                    tf.nn.sigmoid_cross_entropy_with_logits(\
#                                labels=x, logits=output))

    return {
        'W': W,
        'encoded': encoded_x,
        'decoded': reconstructed_x,
        'cost' : loss
    }
    
def create2(x, layer_sizes):

    # Build the encoding layers
    dim = layer_sizes[0]
    input_dim = int(x.get_shape()[1])

    # Initialize W using random values in interval [-1/sqrt(n) , 1/sqrt(n)]
    W_enc = tf.Variable(xavier_init([input_dim, dim]))
                    
    # Initialize b to zero
    b_enc = tf.Variable(tf.zeros([dim]))


    rep = tf.nn.sigmoid(tf.matmul(x, W_enc) + b_enc)

    W_dec = tf.transpose(W_enc)
    b_dec = tf.Variable(tf.zeros([dim]))
    dec = tf.matmul(rep, W_dec) + b_dec
    rec = tf.nn.sigmoid(dec)
    
    loss = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(rec, x), 2.0))
#    tf.reduce_sum(\
#                    tf.nn.sigmoid_cross_entropy_with_logits(\
#                                labels=x, logits=output))

    return {
        'decoded': rec,
        'cost' : loss
    }
    
def plot(inp, samples):
    fig = plt.figure(figsize=(6, 2))
    gs = gridspec.GridSpec(6, 2)
    gs.update(wspace=0.05, hspace=0.05)

    itr = 0
    for x_img, sample in zip(inp, samples):
        ax = plt.subplot(gs[itr])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(x_img.reshape(28, 28), cmap='Greys_r')
        ax = plt.subplot(gs[itr+1])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
        itr += 2
        if itr >= 12:
            break

    return fig

def deep_test():
    MB_SIZE = 64
    MAX_ITER = 10000
    
    mnist = input_data.read_data_sets('/home/kashefy/data/MNIST_data', one_hot=True)
    x = tf.placeholder("float", [None, mnist.train.images.shape[1]], name='x')
    ae = create2(x, [256,128])
    ae_solver = tf.train.GradientDescentOptimizer(0.05).minimize(ae['cost'])
        
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        
        for itr in xrange(MAX_ITER):
            sess.run(init)
            x_mb, _ = mnist.train.next_batch(MB_SIZE)
            _, ae_loss = sess.run([ae_solver, ae['cost']],
                                  feed_dict={x: x_mb})
#            print ae_loss
            if itr % 1000 == 0:
                samples = sess.run(ae['decoded'],
                                   feed_dict={x: x_mb})
                print x_mb[:,784/2-3:784/2+3]
                print samples[:,784/2-3:784/2+3]
                fig = plot(x_mb, samples)
                plt.savefig('/home/kashefy/models/ae/out/{}.png'.format(str(itr).zfill(5)), bbox_inches='tight')
                plt.close(fig)

if __name__ == '__main__':
    
    deep_test()
    pass
