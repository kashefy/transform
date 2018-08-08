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
from nideep.eval.metric_tf import resettable_metric
from nideep.datasets.mnist.mnist_tf import MNIST # TODO remove
from transform.augmentation import rotation_rad
from tensorflow.contrib.learn.python.learn.datasets.mnist import dense_to_one_hot

MetricOps = collections.namedtuple('MetricOps', ['metric', 'update', 'reset'])

class MLP2TaskRunner(AbstractRunner):
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
        if not self.do_finetune:
            vars_new = self.model.vars_new()
            self.init_vars(sess, vars_new)
        summaries_merged_train = self._merge_summaries_scalars([cost, loss])
        
        if self._acc_ops is None:
            self._acc_ops =  self._init_acc_ops()
        sess.run(self._acc_ops.reset)
        if self._acc_orient_ops is None:
            self._acc_orient_ops =  self._init_acc_orient_ops()
        sess.run(self._acc_orient_ops.reset)
        summaries_merged_val = self._merge_summaries_scalars([self._acc_ops.metric,
                                                              self._acc_orient_ops.metric])
#        
#        in_ = self.model.x
#        xx = tf.placeholder("float", [None, 784])
#        augment_op = augment_rotation(xx,
#                                      -90, 90, 15,
#                                      self.batch_size_train)
#        self.model.x = augment_op
        if self.tf_record_prefix is not None:
            img, label, label_orient = MNIST.read_and_decode_ops(\
                                self.data.train.path,
                                one_hot=self.data.train.one_hot,
                                num_orientations=len(self.data.train.orientations))
            batch_xs_op, batch_ys_op, batch_os_op = tf.train.shuffle_batch([img, label, label_orient],
                                                    batch_size=self.batch_size_train,
                                                    capacity=2000,
                                                    min_after_dequeue=1000,
                                                    num_threads=8
                                                    )
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        if self.do_augment_rot:
            rots = rotation_rad(-60, 60, 15)
        self._init_saver()
        itr_exp = 0
        result = collections.namedtuple('Result', ['max', 'last', 'name', 'history', 'epoch_last'])
        result_orient = collections.namedtuple('Result', ['max', 'last', 'name', 'history', 'epoch_last'])
        result.name = self._acc_ops.metric.name
        result.max = 0
        result.history = collections.deque(maxlen=3)
        result_orient.name = self._acc_orient_ops.metric.name
        result_orient.max = 0
        result_orient.history = collections.deque(maxlen=3)
        for epoch in xrange(self.training_epochs):
            self.logger.info("Start %s epoch %d, step %d" % (suffix, epoch, itr_exp))
            # Loop over all batches
            for itr_epoch in xrange(self.num_batches_train):
                if self.tf_record_prefix is None:
                    batch_xs, batch_ys = self.data.train.next_batch(self.batch_size_train)
                else:
                    batch_xs, batch_ys, batch_os = sess.run([batch_xs_op, batch_ys_op, batch_os_op])
#                f = sess.run([augment_op], feed_dict={xx:batch_xs})
                batch_xs_in = batch_xs
                if self.do_augment_rot:
                    augment_op, batch_os2 = self.rotation_ops_multiset_train(3)
                    batch_xs_in = sess.run(augment_op, feed_dict={self.x : batch_xs})
                    orients_dense = np.array([rots.index(o) for o in batch_os2])
                    batch_os_one_hot = dense_to_one_hot(orients_dense, len(rots))
                _, _, sess_summary = sess.run([optimizer,
                                               cost,
                                               summaries_merged_train],
                                               feed_dict={self.x : batch_xs_in,
                                                          self.y_: batch_ys,
                                                          self.orient_ : batch_os_one_hot}
                                              )
                if self.is_time_to_track_train(itr_exp):
                    summary_writer_train.add_summary(sess_summary, itr_exp)
#                self.logger.debug("training batch loss after step %d: %f" % (itr_exp, loss_batch))
                itr_exp += 1
            self.validate(sess)
            # run metric op one more time, data in feed dict is dummy data, does not influence metric
            acc, acc_orient, sess_summary = sess.run([
                                        self._acc_ops.metric,
                                        self._acc_orient_ops.metric,
                                        summaries_merged_val],
                                        feed_dict={self.x  : batch_xs,
                                                    self.y_ : batch_ys,
                                                    self.orient_ : batch_os_one_hot}
                                         )
            if self.is_time_to_track_val(itr_exp):
                summary_writer_val.add_summary(sess_summary, itr_exp)
            self.logger.debug("validation accuracy after %s step %d: %f" % (suffix, itr_exp, acc))
            self.logger.debug("validation orientation accuracy after %s step %d: %f" % (suffix, itr_exp, acc_orient))
            fpath_save = os.path.join(dir_train, self._get_save_name())
            self.logger.debug("Save model at %s step %d to '%s'" % (suffix, itr_exp, fpath_save))
            self.saver.save(sess, fpath_save, global_step=itr_exp)
            result.last = acc
            result.epoch_last = epoch
            result.max = max(result.max, result.last)
            result.history.append(result.last)
            result_orient.last = acc_orient
            result_orient.max = max(result_orient.max, result_orient.last)
            result_orient.history.append(result_orient.last)
            result_orient.epoch_last = epoch
            if self.do_task_recognition:
                if len(result.history) == result.history.maxlen and np.absolute(np.mean(result.history)-result.last) < 1e-5:
                    self.logger.debug("Validation accuracy not changing anymore. Stop iterating.")
                    break
            elif self.do_task_orientation:
                if len(result_orient.history) == result_orient.history.maxlen and np.absolute(np.mean(result_orient.history)-result_orient.last) < 1e-5:
                    self.logger.debug("Validation orientation accuracy not changing anymore. Stop iterating.")
                    break
        if self.tf_record_prefix is not None:
            coord.request_stop()
            coord.join(threads)
        self.logger.info("Classification %s Optimization Finished!" % suffix)
        return result, result_orient
        
    def validate(self, sess):
        if self._acc_ops is None:
            self._acc_ops =  self._init_acc_ops()
        if self._acc_orient_ops is None:
            self._acc_orient_ops =  self._init_acc_orient_ops()
        sess.run(self._acc_ops.reset)
        sess.run(self._acc_orient_ops.reset)
        num_batches_val = int(self.data.validation.num_examples/self.batch_size_val)
        if self.tf_record_prefix is not None:
#            tmp = sum(1 for _ in tf.python_io.tf_record_iterator(self.data.validation.path))
#            assert(num_batches_val == tmp)
            img, label, label_orient = MNIST.read_and_decode_ops(\
                                self.data.validation.path,
                                one_hot=self.data.validation.one_hot,
                                num_orientations=len(self.data.validation.orientations))
            batch_xs_op, batch_ys_op, batch_os_op = tf.train.batch([img, label, label_orient],
                                                    batch_size=self.batch_size_val,
                                                    capacity=2000,
                                                    num_threads=8
                                                    )
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            
        for _ in xrange(num_batches_val):
            if self.tf_record_prefix is None:
                batch_xs, batch_ys = self.data.validation.next_batch(self.batch_size_val,
                                                                     shuffle=False)
            else:
                batch_xs, batch_ys, batch_os = sess.run([batch_xs_op, batch_ys_op, batch_os_op])
            batch_xs_in = batch_xs
            if self.do_augment_rot:
                augment_op, batch_os2 = self.rotation_ops_multiset_val(3)
                rots = rotation_rad(-60,60,15)
                num_orients = len(rots)
                orients_dense = np.array([rots.index(o) for o in batch_os2])
                batch_os_one_hot = dense_to_one_hot(orients_dense, num_orients)
                batch_xs_in = sess.run(augment_op, feed_dict={self.x : batch_xs})
            _, _, _, _ = sess.run(\
                            [self._acc_ops.metric, self._acc_ops.update,
                             self._acc_orient_ops.metric, self._acc_orient_ops.update,
#                             tf.argmax(self.model.p,1), tf.argmax(self.y_,1),
                             ],
                            feed_dict={self.x: batch_xs_in,
                                       self.y_: batch_ys,
                                       self.orient_ : batch_os_one_hot,
                                       }
                            )
        if self.tf_record_prefix is not None:
            coord.request_stop()
            coord.join(threads)
        
    def _cost_loss(self, prefix):
        loss_c = None
        if self.do_task_recognition:
            loss_c = self.lambda_c_recognition * self.model.cost(self.y_, name=prefix + '/loss_classification')
            loss = loss_c
        if self.do_task_orientation:
            loss_o = self.lambda_c_orientation * self.model.cost_aux(self.orient_, name=prefix + '/loss_orientation')
            if loss_c is not None:
                loss = tf.add(loss_c, loss_o, name=prefix + '/loss')
            else:
                loss = loss_o
        regularization_l2 = None
        if self.lambda_l2 != 0:
            regularization_l2 = self._regularization(name=prefix + '/regularization_l2')
        regularization_l1 = None
        if self.lambda_l1 != 0:
            regularization_l1 = self._regularization_l1(name=prefix + '/regularization_l1')
        if regularization_l2 is None and regularization_l1 is None:
            cost = loss
        elif regularization_l2 is not None and regularization_l1 is None:
            cost = tf.add(loss, self.lambda_l2 * regularization_l2,
                  name=prefix + '/cost')
        elif regularization_l2 is None and regularization_l1 is not None:
            cost = tf.add(loss, self.lambda_l1 * regularization_l1,
                  name=prefix + '/cost')
        else:
            cost = tf.add(loss, self.lambda_l1 * regularization_l1 + self.lambda_l2 * regularization_l2,
                  name=prefix + '/cost')
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
    
    def _init_acc_orient_ops(self, name='acc_orient'):
        y_class_op = tf.argmax(self.orient_, 1)
        predicted_class_op = tf.argmax(self.model.p_aux, 1)
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
        super(MLP2TaskRunner, self).__init__(params)
        self.data = self._init_data_mnist()
        self.num_batches_train = int(self.data.train.num_examples/self.batch_size_train)
        self.logger.debug("No. of training batches per epoch: %d" % self.num_batches_train)
        self._check_validation_batch_size()
        self.y_ = None
        self.orient_ = None
        self._acc_ops = None
        self._acc_orient_ops = None
        self.do_finetune = False
        self.do_task_recognition = params['do_task_recognition']
        self.logger.debug("Target recognition: %s" % (['No', 'Yes'][self.do_task_recognition],))
        self.do_task_orientation = params['do_task_orientation']
        self.logger.debug("Target orientation: %s" % (['No', 'Yes'][self.do_task_orientation],))
        assert(self.do_task_recognition or self.do_task_orientation)
        if self.do_task_orientation:
            assert(self.do_augment_rot)
        self.lambda_c_recognition = params['lambda_c_recognition']
        self.logger.debug("Lambda for recognition: %f" % self.lambda_c_recognition)
        self.lambda_c_orientation = params['lambda_c_orientation']
        self.logger.debug("Lambda for orientation #1: %f" % self.lambda_c_orientation)
        
        
