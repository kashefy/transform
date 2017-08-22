'''
Created on Jul 19, 2017

@author: kashefy
'''
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

class MLPRunner(AbstractRunner):
    '''
    classdocs
    '''
    def learn(self, sess):
        dir_train = os.path.join(self.run_dir, self.prefix, 'train')
        summary_writer_train = tf.summary.FileWriter(dir_train,
                                                     sess.graph)
        dir_val = os.path.join(self.run_dir, self.prefix, 'validation')
        summary_writer_val = tf.summary.FileWriter(dir_val)
        self.y_ = tf.placeholder("float", [None, self.model.n_nodes[-1]])
        
        loss = self.model.cost(self.y_, name=self.prefix + "/train/loss_classification")
        if self.lambda_l2 != 0:
            regularization = self._regularization(name=self.prefix + '/train/regularization_l2')
            cost = tf.add(loss, self.lambda_l2 * regularization,
                          name=self.prefix + 'train/cost')
        else:
            cost = loss
        vars_new = self.model.vars_new()
        optimizer = setup_optimizer(cost, self.learning_rate, var_list=vars_new)
        vars_new = self.model.vars_new()
        self.init_vars(sess, vars_new)
        summaries_merged_train = self._merge_summaries_scalars([loss, cost])
        
        if self._acc_ops is None:
            self._acc_ops =  self._init_acc_ops()
        sess.run(self._acc_ops.reset)
        summaries_merged_val = self._merge_summaries_scalars([self._acc_ops.metric])
            
        self._init_saver()
        itr_exp = 0
        for epoch in xrange(self.training_epochs):
            self.logger.info("Start epoch %d, step %d" % (epoch, itr_exp))
            # Loop over all batches
            for itr_epoch in xrange(self.num_batches_train):
                batch_xs, batch_ys = self.data.train.next_batch(self.batch_size_train)
                _, _, sess_summary = sess.run([optimizer,
                                               cost,
                                               summaries_merged_train],
                                               feed_dict={self.x : batch_xs,
                                                          self.y_: batch_ys}
                                              )
                summary_writer_train.add_summary(sess_summary, itr_exp)
#                self.logger.debug("training batch loss after step %d: %f" % (itr_exp, loss_batch))
                itr_exp += 1
            self.validate(sess)
            # run metric op one more time, data in feed dict is dummy data, does not influence metric
            acc, sess_summary = sess.run([self._acc_ops.metric, summaries_merged_val],
                                         feed_dict={self.x  : batch_xs,
                                                    self.y_ : batch_ys}
                                         )
            summary_writer_val.add_summary(sess_summary, itr_exp)
            self.logger.debug("validation accuracy after step %d: %f" % (itr_exp, acc))
            fpath_save = os.path.join(dir_train, self._get_save_name())
            self.logger.debug("Save model at step %d to '%s'" % (itr_exp, fpath_save))
            self.saver.save(sess, fpath_save, global_step=itr_exp)  
        self.logger.info("Classification Optimization Finished!")
        
    def validate(self, sess):
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
        self.logger.debug("No. of training batches per epoch:", self.num_batches_train)
        self._check_validation_batch_size()
        self.y_ = None
        self._acc_ops = None
        self.prefix = 'classification'
        