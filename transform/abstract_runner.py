'''
Created on Jul 19, 2017

@author: kashefy
'''
import os
import logging
from abc import ABCMeta, abstractmethod
import yaml
import zlib
from bunch import Bunch
import numpy as np
import tensorflow as tf
from nideep.datasets.mnist.mnist_tf import MNIST
from augmentation import rotation_ops
from transform.augmentation import gaussian_noise_op, rotation_rad

def setup_optimizer(cost, learning_rate, name=None, var_list=None):
#    opt = tf.train.RMSPropOptimizer(learning_rate)
    if name is None:
        op = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        op = tf.train.GradientDescentOptimizer(learning_rate,
                                               name=name)
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
    
    def init_vars(self, sess, vars_):
        vars_restored_names = [v.name for v in self.model.vars_restored]
        vars_sub = [v for v in vars_ if v.name not in vars_restored_names]
        init_op = tf.variables_initializer(vars_sub)
        self.logger.debug('initializing: %s' % [v.name for v in vars_sub])
        sess.run(init_op)
        
    def dirpath(self, phase, suffix=''):
        return os.path.join(self.run_dir,
                            self.dirname(phase, suffix=suffix))
        
    def dirname(self, phase, suffix=''):
        return os.path.join(self.prefix,
                             '_'.join([phase, suffix]).rstrip('_'))
        
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
    
    def _merge_summaries_scalars(self, ops):
        summaries = []
        for value in ops:
            self.logger.debug("log scalar: %s" % value.op.name)
            summaries.append(tf.summary.scalar(value.op.name,
                                               value))
        return tf.summary.merge(summaries)
    
    def _append_summaries_hists(self, summaries, t):
        for value in t:
            self.logger.debug("log histograms: %s" % value.name)
            hist = tf.summary.histogram(value.name.replace(':', '_'),
                                        value)
            summaries.append(hist)
        
    def _append_summaries_var_imgs(self, summaries, vars):
        for value in vars:
            warn_reshape_done = False
            name_prefix_ = value.name.replace(':', '_')
            self.logger.debug("log images: %s" % value.name)
            dim_, num_filters = value.get_shape()
            for fidx in xrange(num_filters):
                name_ = name_prefix_ + '_%d' % fidx
                img_vals = value[:, fidx]
                if np.sqrt(dim_.value) ** 2 == dim_.value:
                    img = tf.reshape(img_vals,
                                     [1,
                                      int(np.sqrt(dim_.value)),
                                      int(np.sqrt(dim_.value)),
                                      1])
                    summaries.append(tf.summary.image(name_, img))
                elif not warn_reshape_done:
                    self.logger.debug("Cannot reshape %s %s to a square image. Skipping." % (name_, dim_))
                    warn_reshape_done = True
                    
    def _append_summaries_op_imgs(self, summaries, t):
        for op in t:
            name_prefix_ = op.name.replace(':', '_')
            _, dim_ = op.get_shape()
            self.logger.debug("log images: %s" % op.name)
            if np.sqrt(dim_.value) ** 2 == dim_.value:
                p_img = tf.reshape(op,
                                   [self.batch_size_val,
                                    int(np.sqrt(dim_.value)),
                                    int(np.sqrt(dim_.value)),
                                    1])
                summaries.append(tf.summary.image(self.model.p.name.replace(':', '_'), p_img))
            else:
                self.logger.debug("Cannot reshape %s %s to a square image. Skipping." % (name_prefix_, dim_))
    
    def _init_saver(self, force=False, var_list=None):
        if self.saver is None:
            force = True
        if force:
            self.saver = tf.train.Saver(var_list=var_list,
                                        max_to_keep=5)
            
    def _get_save_name(self):
        return '_'.join(['saved', self.model.prefix]).rstrip('_')
    
    def _regularization_l2(self, name=None):
        if self.lambda_l2 != 0:
            weights = self.model.w
            w_names = [weights[k].name for k in weights.keys()]
            self.logger.debug("L2 regularization for %s" % ','.join(w_names))
            losses = [tf.nn.l2_loss(weights[k]) for k in weights.keys()]
            regularizers = tf.add_n(losses, name=name)
            return regularizers
        else:
            return None
        
    def _regularization_l1(self, name=None):
        if self.lambda_l1 != 0:
            weights = self.model.w
            w_names = [weights[k].name for k in weights.keys()]
            self.logger.debug("L1 regularization for %s" % ','.join(w_names))
            losses = [tf.norm(weights[k], ord=1) for k in weights.keys()]
            regularizers = tf.add_n(losses, name=name)
            return regularizers
        else:
            return None
        
    def _regularization_l2(self, name=None):
        if self.lambda_l2 != 0:
            weights = self.model.w
            w_names = [weights[k].name for k in weights.keys()]
            self.logger.debug("L2 regularization for %s" % ','.join(w_names))
            losses = [tf.nn.l2_loss(weights[k]) for k in weights.keys()]
            regularizers = tf.add_n(losses, name=name)
            return regularizers
        else:
            return None
    
    def _check_validation_batch_size(self):
        num_batches_val_real = self.data.validation.num_examples/float(self.batch_size_val)
        num_batches_val_trunc = int(self.data.validation.num_examples/self.batch_size_val)
        if num_batches_val_real > num_batches_val_trunc:
            self.logger.warning("Validation batch size of will shift data every epoch. "
                                "%d * %d = %d out of %d" % (self.batch_size_val,
                                                            num_batches_val_trunc,
                                                            self.batch_size_val * num_batches_val_trunc,
                                                            self.data.validation.num_examples)
                                )
        else:
            self.logger.debug("Validation data will not shift between epochs.")
            
    def is_time_to_track_train(self, itr):
        return AbstractRunner.is_time_to_track(self.track_interval_train, itr)
        
    def is_time_to_track_val(self, itr):
        return AbstractRunner.is_time_to_track(self.track_interval_val, itr)
        
    @staticmethod
    def is_time_to_track(interval, itr):
        return interval > 0 and itr % interval == 0
    
    def _init_data_mnist(self):
        if self.tf_record_prefix is None:
            data = MNIST.read_data_sets(self.data_dir,
                                        one_hot=True,
                                        validation_size=self.validation_size,
                                        seed=self.data_seed)
        else:
            tf_record_descr = {'data_seed'  : self.data_seed,
                               'one_hot'    : True,
                               'orientations' : sorted(np.rad2deg(rotation_rad(-60,60,15))),
                               'validation_size' : self.validation_size
                               }
            descr_str = '_'.join(['-'.join([k, str(tf_record_descr[k])])
                                  for k in sorted(tf_record_descr.keys())])
            descr_hash = zlib.adler32(descr_str)
            self.logger.debug("TF Record description (hash:%s):'%s'" % (descr_hash, descr_str))
            tf_record_name = '%s_%s' % (self.tf_record_prefix, descr_hash)
            fpath_tf_record_descr = os.path.join(self.data_dir, tf_record_name + '.yml')
            self.logger.debug("Save TF Record description (hash:%s) to %s" % (descr_hash, fpath_tf_record_descr))
            with open(fpath_tf_record_descr, 'w') as h:
                h.write(yaml.dump(tf_record_descr))
            data = MNIST.to_tf_record(os.path.join(self.data_dir, tf_record_name + '.tfrecords'),
                           self.data_dir,
                           one_hot=tf_record_descr['one_hot'],
                           orientations=tf_record_descr['orientations'],
                           seed=tf_record_descr['data_seed'])
            self.logger.info("Data will be loaded from TF Records: "
                             "%s" % ', '.join([':'.join([f,str(getattr(data, f).path)]) for f in data._fields])
                             )
        return data
    
    def rotation_ops_multiset_train(self,
                                    count):
        if self.augment_ops_train is None:
            self.augment_ops_train = []
            for _ in range(count):
                augment_op, rotations = rotation_ops(self.x,
                                      -60, 60, 15,
                                      self.batch_size_train,
                                      self.prefix)
                self.augment_ops_train.append((augment_op, rotations))
        augment_op = self.augment_ops_train[np.random.randint(count)]
        return augment_op
            
    def rotation_ops_multiset_val(self,
                                    count):
        if self.augment_ops_val is None:
            self.augment_ops_val = []
            for _ in range(count):
                augment_op, rotations = rotation_ops(self.x,
                                      -60, 60, 15,
                                      self.batch_size_val,
                                      self.prefix)
                self.augment_ops_val.append((augment_op, rotations))
        augment_op = self.augment_ops_val[np.random.randint(count)]
        return augment_op
    
    def gaussian_noise_op(self, in_):
        op_name = '_'.join([in_.name, 'noise_add'])
        if op_name in self.gaussian_noise_ops:
            return self.gaussian_noise_ops[op_name]
        else:
            op_new = gaussian_noise_op(in_, self.input_noise_std)
            assert(op_new.name == op_name)
            self.gaussian_noise_ops[op_name] = op_new
        return op_new
    
    def __init__(self, params):
        '''
        Constructor
        '''
        params = Bunch(params)
        logger_name = self.__class__.__name__
        if params.logger_name is not None:
            logger_name = '.'.join([params.logger_name, logger_name])
        self.logger = logging.getLogger(logger_name)
        self.run_name = params.run_name
        self.run_dir = params.run_dir
        
        self.prefix = params.prefix
        self.logger.debug("Runner prefix: '%s'" % self.prefix)
        self.batch_size_train = params.batch_size_train
        self.logger.debug("batch_size (training): %d" % self.batch_size_train)
        self.batch_size_val = params.batch_size_val
        self.logger.debug("batch_size (validation): %d" % self.batch_size_val)
        self.learning_rate = params.learning_rate
        self.logger.debug("initial learning rate: %g" % self.learning_rate)
        self.training_epochs = params.training_epochs
        self.logger.debug("training epochs: %d" % self.training_epochs)
        self.validation_size = params.validation_size
        self.logger.debug("validation size: %d" % self.validation_size)
        self.lambda_l2 = params.lambda_l2
        if self.lambda_l2 != 0:
            self.logger.debug("lambda_l2: %g" % self.lambda_l2)
        else:
            self.logger.debug("No L2 regularization")
        self.lambda_l1 = params.lambda_l1
        if self.lambda_l1 != 0:
            self.logger.debug("lambda_l1: %g" % self.lambda_l1)
        else:
            self.logger.debug("No L1 regularization")
        self._model = None
        self.saver = None
        self.track_interval_train = params.track_interval_train
        self.track_interval_val = params.track_interval_val
        self.data_dir = params.data_dir
        self.logger.debug("Data dir: %s" % self.data_dir)
        self.data_seed = params.data_seed
        if self.data_seed is None:
            self.logger.debug("Generate new data seed")
        else:
            self.logger.debug("Data seed: %d" % self.data_seed)
        self.tf_record_prefix = params.tf_record_prefix
        if self.tf_record_prefix is not None:
            self.logger.debug("Use tf records prefix '%s'" % self.tf_record_prefix)
        else:
            self.logger.debug("No tf records prefix provided. Don't load data from tf records")
        self.do_augment_rot = params.do_augment_rot
        self.logger.debug("Augment by rotation: %s" % (['No', 'Yes'][self.do_augment_rot],))
        self.input_noise_std = params['input_noise_std']
        self.logger.debug("Input noise std-dev: %f" % self.input_noise_std,)
        self.augment_ops_train = None
        self.augment_ops_val = None
        self.gaussian_noise_ops = {}