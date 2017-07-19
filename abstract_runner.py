'''
Created on Jul 19, 2017

@author: kashefy
'''
import os
import logging
from abc import ABCMeta, abstractmethod
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
    
    def __init__(self, params):
        '''
        Constructor
        '''
        self.run_name = params['run_name']
        self.run_dir = os.path.join(params.log_dir, self.run_name)
        self.logger = logging.getLogger(__name__)
        
        self.batch_size = params.batch_size
        self.learning_rate = params.learning_rate
        self.training_epochs = params.training_epochs