'''
Created on Jul 19, 2017

@author: kashefy
'''
import os
import logging
from abc import ABCMeta, abstractmethod
from bunch import Bunch
import tensorflow as tf

def setup_optimizer(cost, learning_rate, var_list=None):
#    opt = tf.train.RMSPropOptimizer(learning_rate)
    op = tf.train.GradientDescentOptimizer(learning_rate)
    op_min = op.minimize(cost, var_list=var_list)
    return op_min

class AbstractRunner(object):
    '''
    classdocs
    '''
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def learn(self):
        pass
    
    def init_vars(self, sess, vars):
        init_op = tf.variables_initializer(vars)
        self.logger.debug('initializing %s' % [v.name for v in vars])
        sess.run(init_op)
        
    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    @model.deleter
    def model(self):
        del self._model
        
    def _init_learning_params(self):
        pass
    
    def _merge_summaries(self, ops):
        summaries = []
        for value in ops:
            self.logger.debug("log scalar: %s" % value.op.name)
            summaries.append(tf.summary.scalar(value.op.name, value))
        return tf.summary.merge(summaries)
    
    def _init_saver(self, force=False):
        if self.saver is None:
            force = True
        if force:
            self.saver = tf.train.Saver(max_to_keep=5)
            
    def _get_save_name(self):
        return '_'.join(['saved', self.model.prefix]).rstrip('_')
    
    def _regularization(self, name=None):
        if self.lambda_l2 != 0:
            weights = self.model.w
            w_names = [weights[k].name for k in weights.keys()]
            self.logger.debug("L2 regularization for %s" % ','.join(w_names))
            losses = [tf.nn.l2_loss(weights[k]) for k in weights.keys()]
            regularizers = tf.add_n(losses, name=name)
            return regularizers
        else:
            return None
    
    def __init__(self, params):
        '''
        Constructor
        '''
        params = Bunch(params)
        self.run_name = params.run_name
        if params.logger_name is None:
            logger_name = __name__
        else:
            logger_name = '.'.join([params.logger_name, __name__])
        self.logger = logging.getLogger(logger_name)
        self.run_dir = os.path.join(params.log_dir, self.run_name)
        
        self.prefix = params.prefix
        self.batch_size = params.batch_size
        self.logger.debug("batch_size: %d", self.batch_size)
        self.learning_rate = params.learning_rate
        self.logger.debug("initial learning rate: %f", self.learning_rate)
        self.training_epochs = params.training_epochs
        self.logger.debug("training epochs: %d", self.training_epochs)
        self.validation_size = params.validation_size
        self.logger.debug("validation size: %d", self.validation_size)
        self.lambda_l2 = params.lambda_l2
        if self.lambda_l2 != 0:
            self.logger.debug("lambda_l2: %f", self.lambda_l2)
        else:
            self.logger.debug("No L2 regularization")
        self._model = None
        self.saver = None
        