'''
Created on May 26, 2017

@author: kashefy
'''
from __future__ import division, print_function, absolute_import
import tensorflow as tf

class Autoencoder(object):
    '''
    classdocs
    '''
    @staticmethod
    def _init_weight_op(sh):
        return tf.random_normal(sh)
    
    @staticmethod
    def _init_bias_op(n):
        return tf.random_normal([n])
    
    def _encoder_w_name(self):
        return 'encoder_h_%d' % self.depth
    
    def _decoder_w_name(self):
        return 'decoder_h_%d' % self.depth
    
    def _encoder_b_name(self):
        return 'encoder_b_%d' % self.depth
    
    def _decoder_b_name(self):
        return 'decoder_b_%d' % self.depth
    
    def _representation_name(self):
        return 'z_%d' % self.depth
    
    def _reconstruction_name(self):
        return "h_%d" % self.depth
    
    def _init_learning_params(self):
        # tf Graph input (only pictures)
        with tf.name_scope(self.name_scope) as _:
            encoder_name = self._encoder_w_name()
            self.w = {
                encoder_name: tf.Variable(self._init_weight_op([self.n_input,
                                                               self.n_hidden_1]),
                                          name=encoder_name),
            }
            decoder_name = self._decoder_w_name()
            self.w[decoder_name] = tf.transpose(self.w[encoder_name], name=decoder_name)
            
            self.b = {}
            for key, value in self.w.iteritems():
                key_b = key.replace('_h', '_b')
                b_dim = int(value.get_shape()[-1])
                self.b[key_b] = tf.Variable(self._init_bias_op(b_dim), name=key_b)
                
    def representation(self):
        return self._representation_op

    def encoder(self, x):
        # Building the encoder
        # Encoder Hidden layer with sigmoid activation #1
        self._representation_op = tf.nn.sigmoid(tf.add(tf.matmul(x, self.w[self._encoder_w_name()]),
                                                    self.b[self._encoder_b_name()]),
                                                name=self._representation_name())
        return self._representation_op
        
    def decoder(self, z):
        # Encoder Hidden layer with sigmoid activation #1
        self._decoder_logits_op = tf.add(tf.matmul(z, self.w[self._decoder_w_name()]),
                                         self.b[self._decoder_b_name()])
        self.y_pred = tf.nn.sigmoid(self._decoder_logits_op,
                                    name=self._reconstruction_name())
        return self.y_pred, self._decoder_logits_op
    
    def construct_model(self):
        # Construct model
        encoder_op = self.encoder(self.x)
        decoder_reconstruction_op, decoder_logits_op = self.decoder(encoder_op)
        return decoder_reconstruction_op, decoder_logits_op
    
    def cost_euclidean(self, y_true):
        return tf.reduce_mean(tf.pow(y_true - self.y_pred, 2))

    def cost_cross_entropy(self, y_true):
        return tf.reduce_mean(\
                  tf.nn.sigmoid_cross_entropy_with_logits(\
                      labels=y_true, logits=self._decoder_logits_op))
        
    @property
    def x(self):
        """I'm the 'x' property."""
        return self._x

    @x.setter
    def x(self, value):
        self._x = value

    @x.deleter
    def x(self):
        del self._x
        
    @property
    def y_pred(self):
        return self._y_pred

    @y_pred.setter
    def y_pred(self, value):
        self._y_pred = value

    @y_pred.deleter
    def y_pred(self):
        del self._y_pred
    
    def __init__(self, params):
        '''
        Constructor
        '''
        self._x = None
        self._y_pred = None
        
        # Network Parameters
        self.n_hidden_1 = params['n_hidden_1']  # 1st layer num features
        self.n_input = params['n_input'] # MNIST data input (img shape: 28*28)
        if 'depth' not in params:
            depth = 1
        else:
            depth = params['depth']
        self.depth = depth
        self.name_scope = 'layer_%d' %  self.depth
        self._init_learning_params()
        
            
        
