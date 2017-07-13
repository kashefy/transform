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
    def _init_weight_op():
        return tf.random_normal_initializer()
    
    @staticmethod
    def _init_bias_op(value):
        return tf.constant_initializer(value)
    
    def _encoder_w_name(self):
        return 'encoder-%d/w' % self.depth
    
    def _decoder_w_name(self):
        return 'decoder-%d/w' % self.depth
    
    def _encoder_b_name(self):
        return 'encoder-%d/b' % self.depth
    
    def _decoder_b_name(self):
        return 'decoder-%d/b' % self.depth
    
    def _representation_name(self):
        return 'z-%d' % self.depth
    
    def _reconstruction_name(self):
        return "hd-%d" % self.depth
    
    def _init_learning_params(self):
        with tf.variable_scope(self.var_scope):
            encoder_name = self._encoder_w_name()
            self.w = {
                encoder_name: tf.get_variable(encoder_name,
                                              [self.n_input, self.n_hidden_1],
                                              initializer=self._init_weight_op(),
                                              )
            }
            decoder_name = self._decoder_w_name()
            self.w[decoder_name] = tf.transpose(self.w[encoder_name],
                                                name=decoder_name)
            self.b = {}
            for key, value in self.w.iteritems():
                key_b = key.replace('/w', '/b').replace('_w', '_b').replace('-w', '-b')
                self.b[key_b] = tf.get_variable(key_b,
                                                [int(value.get_shape()[-1])],
                                                initializer=self._init_bias_op(0.))
                
    def representation(self):
        return self._representation_op

    def encoder(self, x):
        # Building the encoder
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
        with tf.name_scope(self.name_scope + 'encode'):
            encoder_op = self.encoder(self.x)
        with tf.name_scope(self.name_scope + 'decode'):
            decoder_reconstruction_op, decoder_logits_op = self.decoder(encoder_op)
        return decoder_reconstruction_op, decoder_logits_op
    
    def cost_euclidean(self, y_true):
        with tf.name_scope(self.name_scope):
            if self._cost_op is None:
                self._cost_op = tf.reduce_mean(tf.pow(y_true - self.y_pred, 2))
            return self._cost_op

    def cost_cross_entropy(self, y_true, name=None):
        with tf.name_scope(self.name_scope):
            if self._cost_op is None or True:
                self._cost_op = \
                tf.reduce_mean(\
                      tf.nn.sigmoid_cross_entropy_with_logits(\
                          labels=y_true, logits=self._decoder_logits_op, name=name),
                                  name=name)
            return self._cost_op
        
    @property
    def x(self):
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
        self.n_hidden_1 = params['n_hidden']  # 1st layer num features
        self.n_input = params['n_input'] # MNIST data input (img shape: 28*28)
        if 'depth' not in params:
            depth = 1
        else:
            depth = params['depth']
        self.depth = depth
        self.var_scope = 'layer-%d' %  self.depth
        self.name_scope = self.var_scope + '/'
        self._init_learning_params()
        self._cost_op = None
        
            
        
