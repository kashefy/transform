'''
Created on May 26, 2017

@author: kashefy
'''
from __future__ import division, print_function, absolute_import
from nideep.nets.abstract_net_tf import AbstractNetTF
import tensorflow as tf

class Autoencoder(AbstractNetTF):
    '''
    classdocs
    '''
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
    
    def _init_learning_params_scoped(self):
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
        self.p = tf.nn.sigmoid(self._decoder_logits_op,
                               name=self._reconstruction_name())
        return self.p, self._decoder_logits_op

    def build(self):
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
    
    def __init__(self, params):
        '''
        Constructor
        '''
        
        # Network Parameters
        self.n_hidden_1 = params['n_hidden']  # 1st layer num features
        self.n_input = params['n_input'] # MNIST data input (img shape: 28*28)

        super(Autoencoder, self).__init__(params)
        self._cost_op = None
        
            
        
