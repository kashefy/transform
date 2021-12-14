'''
Created on May 26, 2017

@author: kashefy
'''
from __future__ import division, print_function, absolute_import
from nideep.nets.abstract_net_tf import AbstractNetTF
import tensorflow as tf
from transform.augmentation import gaussian_noise_op

class Autoencoder(AbstractNetTF):
    '''
    classdocs
    '''
    def _encoder_w_name(self, idx):
        return 'encoder-%d/w' % idx
    
    def _decoder_w_name(self, idx):
        return 'decoder-%d/w' % idx
    
    def _encoder_b_name(self, idx):
        return 'encoder-%d/b' % idx
    
    def _decoder_b_name(self, idx):
        return 'decoder-%d/b' % idx
    
    def _representation_name(self, idx):
        return 'z-%d' % idx
    
    def _reconstruction_name(self, idx):
        return "hd-%d" % idx
    
    def _init_learning_params_scoped(self):
        self.w = {}
        for idx, dim in enumerate(self.n_nodes):
            input_dim = self.n_nodes[idx-1]
            if idx == 0:
                input_dim = self.n_input
            encoder_w_name = self._encoder_w_name(idx)
            decoder_w_name = self._decoder_w_name(idx)
            if self.reuse:
                self.w[encoder_w_name] = self._restore_variable(self.var_scope + '/' + encoder_w_name)
                self.w[decoder_w_name] = self._restore_variable(self.var_scope + '/' + decoder_w_name)
            else:
                self.w[encoder_w_name] = tf.get_variable(encoder_w_name,
                                                         [input_dim, dim],
                                                         initializer=self._init_weight_op(),
                                                         )
                decoder_w_name = self._decoder_w_name(idx)
                self.w[decoder_w_name] = tf.transpose(self.w[encoder_w_name],
                                                      name=decoder_w_name)
        self._init_bias_vars()
            
    def representation(self):
        return self._representation_op

    def _encoder_op(self, x, idx):
        op = tf.nn.sigmoid(tf.add(tf.matmul(x, self.w[self._encoder_w_name(idx)]),
                                  self.b[self._encoder_b_name(idx)]),
                           name=self._representation_name(idx))
        return op
    
    def _decoder_op(self, z, idx):
        # Encoder Hidden layer with sigmoid activation
        logits_op = tf.add(tf.matmul(z, self.w[self._decoder_w_name(idx)]),
                           self.b[self._decoder_b_name(idx)])
        op = tf.nn.sigmoid(logits_op, name=self._reconstruction_name(idx))
        return op, logits_op

    def decoder(self, z):
        # Encoder Hidden layer with sigmoid activation
        self.p, self._decoder_logits_op = \
            self._decoder_op(z, len(self.n_nodes)-1)
        return self.p, self._decoder_logits_op

    def _init_ops(self):
        # Construct model
        encoder_op = self.x
        for idx in range(len(self.n_nodes)):
            with tf.name_scope(self.name_scope + 'encode'):
                if self.do_denoising:
                    encoder_op = self.gaussian_noise_op(encoder_op)
                if idx == 0:
                    self.enc_in = encoder_op
                encoder_op = self._encoder_op(encoder_op, idx)
        self._representation_op = encoder_op
        for idx in range(len(self.n_nodes)-1, -1, -1):
            with tf.name_scope(self.name_scope + 'decode'):
                if idx == len(self.n_nodes)-1:
                    self.p, self._decoder_logits_op = \
                        self.decoder(encoder_op)
                else:
                    self.p, self._decoder_logits_op = \
                        self._decoder_op(self.p, idx)
        self.logits = self._decoder_logits_op
        
    def gaussian_noise_op(self, in_):
        return gaussian_noise_op(in_, self.input_noise_std)
    
    def cost_euclidean(self, y_true, name=None):
        with tf.name_scope(self.name_scope):
            if self._cost_op is None:
                self._cost_op = tf.reduce_mean(tf.pow(y_true - self.p, 2),
                                               name=name)
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
        super(Autoencoder, self).__init__(params)
        self.n_input = params['n_input'] # MNIST data input (img shape: 28*28)
        self.n_nodes = params['n_nodes']  # 1st layer num features
        self._cost_op = None
        self.do_denoising = params.get('do_denoising', False)
        self.input_noise_std = params.get('input_noise_std', 0.)
        if self.input_noise_std == 0.:
            self.do_denoising = False
