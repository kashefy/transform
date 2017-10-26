'''
Created on Jul 19, 2017

@author: kashefy
'''
import numpy as np
import os
import collections
import tensorflow as tf
#from tensorflow.python import debug as tf_debug
from abstract_runner import setup_optimizer
#from kfold_cv_runner import KFoldCVRunner
from abstract_runner import AbstractRunner
from nideep.datasets.mnist.mnist_tf import MNIST
from nideep.eval.metric_tf import resettable_metric

MetricOps = collections.namedtuple('MetricOps', ['metric', 'update', 'reset'])

def rotation_rad(min_deg, max_deg, delta_deg):
    return np.deg2rad(np.arange(min_deg, max_deg+delta_deg, delta_deg)).tolist()

def augment_rotation(x, 
                     min_deg, max_deg, delta_deg,
                     batch_sz
                     ):
    reshape_op = tf.reshape(x, [-1, 28, 28, 1])
    rotations_rad = rotation_rad(min_deg, max_deg, delta_deg)
    rots_cur = np.random.choice(rotations_rad, batch_sz)
    rot_op = tf.contrib.image.rotate(reshape_op, rots_cur)
    flatten_op = tf.reshape(rot_op, [-1, x.get_shape()[-1].value])
    return flatten_op

class MLPRunner(AbstractRunner):
    '''
    classdocs
    '''
    def learn(self, sess):
        if self.y_ is None:
            self.logger.info("Define placeholder for ground truth. Dims: %d" % self.model.n_nodes[-1])
            self.y_ = tf.placeholder("float", [None, self.model.n_nodes[-1]])
        suffix = ''
        if self.do_finetune:
            self.logger.info("Finetuning!")
            suffix += 'finetune'
        dir_train = self.dirpath('train', suffix=suffix)
        dir_val = self.dirpath('validation', suffix=suffix)
            
        summary_writer_train = tf.summary.FileWriter(dir_train,
                                                     sess.graph)
        summary_writer_val = tf.summary.FileWriter(dir_val)
        cost, loss = self._cost_loss(self.dirname('train', suffix=suffix))
        vars_new = None
        if not self.do_finetune:
            vars_new = self.model.vars_new() # limit optimizaer vars if not finetuning
        optimizer = setup_optimizer(cost, self.learning_rate, var_list=vars_new)
        vars_new = self.model.vars_new()
        self.init_vars(sess, vars_new)
        summaries_merged_train = self._merge_summaries_scalars([cost, loss])
        
        if self._acc_ops is None:
            self._acc_ops =  self._init_acc_ops()
        sess.run(self._acc_ops.reset)
        summaries_merged_val = self._merge_summaries_scalars([self._acc_ops.metric])
#        
#        in_ = self.model.x
        xx = tf.placeholder("float", [None, 784])
        augment_op = augment_rotation(xx,
                                      -90, 90, 15,
                                      self.batch_size_train)
#        self.model.x = augment_op
        self._init_saver()
        itr_exp = 0
        result = collections.namedtuple('Result', ['min', 'last'])
        result.min = 9999999
        for epoch in xrange(self.training_epochs):
            self.logger.info("Start %s epoch %d, step %d" % (suffix, epoch, itr_exp))
            # Loop over all batches
            for itr_epoch in xrange(self.num_batches_train):
                batch_xs, batch_ys = self.data.train.next_batch(self.batch_size_train)
#                f = sess.run([augment_op], feed_dict={xx:batch_xs})
                _, _, sess_summary = sess.run([optimizer,
                                               cost,
                                               summaries_merged_train],
                                               feed_dict={self.x : batch_xs,
                                                          self.y_: batch_ys}
                                              )
                if self.is_time_to_track_val(itr_exp):
                    summary_writer_train.add_summary(sess_summary, itr_exp)
#                self.logger.debug("training batch loss after step %d: %f" % (itr_exp, loss_batch))
                itr_exp += 1
            self.validate(sess)
            # run metric op one more time, data in feed dict is dummy data, does not influence metric
            acc, sess_summary = sess.run([self._acc_ops.metric, summaries_merged_val],
                                         feed_dict={self.x  : batch_xs,
                                                    self.y_ : batch_ys}
                                         )
            if self.is_time_to_track_val(itr_exp):
                summary_writer_val.add_summary(sess_summary, itr_exp)
            self.logger.debug("validation accuracy after %s step %d: %f" % (suffix, itr_exp, acc))
            fpath_save = os.path.join(dir_train, self._get_save_name())
            self.logger.debug("Save model at %s step %d to '%s'" % (suffix, itr_exp, fpath_save))
            self.saver.save(sess, fpath_save, global_step=itr_exp)
            result.last = acc
            result.min = min(result.min, acc)
        self.logger.info("Classification %s Optimization Finished!" % suffix)
        return result
        
    def validate(self, sess):
        if self._acc_ops is None:
            self._acc_ops =  self._init_acc_ops()
        sess.run(self._acc_ops.reset)
        num_batches_val = int(self.data.validation.num_examples/self.batch_size_val)
        for _ in xrange(num_batches_val):
            batch_xs, batch_ys = self.data.validation.next_batch(self.batch_size_val,
                                                                 shuffle=False)
            _, _ = sess.run(\
                            [self._acc_ops.metric, self._acc_ops.update,
#                             tf.argmax(self.model.p,1), tf.argmax(self.y_,1),
                             ],
                            feed_dict={self.x: batch_xs,
                                       self.y_: batch_ys}
                            )
        
    def _cost_loss(self, prefix):
        loss = self.model.cost(self.y_, name=prefix + '/loss_classification')
        if self.lambda_l2 != 0:
            regularization = self._regularization(name=prefix + '/regularization_l2')
            cost = tf.add(loss, self.lambda_l2 * regularization,
                          name=prefix + '/cost')
        else:
            cost = loss
        return cost, loss
            
    def _init_acc_ops(self, name='acc'):
        y_class_op = tf.argmax(self.y_, 1)
        predicted_class_op = tf.argmax(self.model.p, 1)
        acc_op, up_op, reset_op = resettable_metric(\
                                                tf.metrics.accuracy,
                                                self.model.name_scope + 'validation',
                                                labels=y_class_op,
                                                predictions=predicted_class_op)
        return MetricOps(metric=acc_op, update=up_op, reset=reset_op)

    def __init__(self, params):
        '''
        Constructor
        '''
        super(MLPRunner, self).__init__(params)
        self.data = MNIST.read_data_sets("MNIST_data",
                                         one_hot=True,
                                         validation_size=self.validation_size)
        self.num_batches_train = int(self.data.train.num_examples/self.batch_size_train)
        self.logger.debug("No. of training batches per epoch: %d" % self.num_batches_train)
        self._check_validation_batch_size()
        self.y_ = None
        self._acc_ops = None
        self.prefix = 'classification'
        self.do_finetune = False
        