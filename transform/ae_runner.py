'''
Created on Jul 19, 2017

@author: kashefy
'''
import os
import collections
import numpy as np
import tensorflow as tf
#from tensorflow.python import debug as tf_debug
from abstract_runner import AbstractRunner, setup_optimizer
from nideep.datasets.mnist.mnist_tf import MNIST # TODO remove

class AERunner(AbstractRunner):
    '''
    classdocs
    '''
    def learn(self, sess):
        dir_train = self.dirpath('train')
        summary_writer_train = tf.summary.FileWriter(dir_train,
                                                     sess.graph)
        dir_val = self.dirpath('validation')
        summary_writer_val = tf.summary.FileWriter(dir_val)
        itr_exp = 0
        result = collections.namedtuple('Result', ['max', 'last', 'name'])
        result.max = 0

#        saver = tf.train.import_meta_graph('/home/kashefy/models/ae/log_simple_stats/pre0_t00/reconstruction/train/saved_sae0-20.meta')
#        saver.restore(sess, tf.train.latest_checkpoint('/home/kashefy/models/ae/log_simple_stats/pre0_t00/reconstruction/train/'))
#        g = tf.get_default_graph()
#        op_names = [op.name for op in g.get_operations()
#                    if op.op_def and 'Variable' in op.op_def.name]
#        for o in op_names:
#            print o
#        print(sess.run('sae0-1/encoder-0/w:0')[10,5:10])
#        print(sess.run('sae0-2/decoder-0/b:0')[:3])
        
#        fpath_save = os.path.join(dir_train, self._get_save_name())
#        self.logger.debug("Save model at step %d to '%s'" % (itr_exp, fpath_save))
#        self.saver.save(sess, fpath_save, global_step=itr_exp)
        if self.tf_record_prefix is not None:
            img, label, label_orient = MNIST.read_and_decode_ops(\
                                self.data.train.path,
                                one_hot=self.data.train.one_hot,
                                num_orientations=len(self.data.train.orientations))
            batch_xs_op, batch_ys_op, batch_os_op = tf.train.shuffle_batch([img, label, label_orient],
                                                    batch_size=self.batch_size_train,
                                                    capacity=2000,
                                                    min_after_dequeue=1000,
                                                    num_threads=4
                                                    )
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for dim_idx, dim in enumerate(self.stack_dims):
            itr_depth = 0
            self.logger.debug('Stacking %s nodes.' % str(dim))
            self.model.stack(dim)
            if itr_exp == 0:
                self.x = self.model.x
#                augment_op = augment_rotation(self.x,
#                                              -90, 90, 15,
#                                              self.batch_size_train)
#                self.model.x = augment_op
                if self.do_reconstruct_original:
                    self.model.y_true = tf.placeholder("float", self.x.get_shape())
            cost, loss = self._cost_loss(self.dirname('train'))
            result.name = loss.name
            vars_new = self.model.vars_new()
            self.logger.debug('Variables added: %s' % [v.name for v in vars_new])
            self._vars_added.append(vars_new)
            optimizer = setup_optimizer(cost, self.learning_rate, var_list=vars_new)
            vars_new = self.model.vars_new()
#            self.logger.debug('encoder-0: %s' % sess.run(self.model.sae[0].w['encoder-0/w'][10,5:10]))
            
            self.init_vars(sess, vars_new)
#            self.logger.debug('encoder-0: %s' % sess.run(self.model.sae[0].w['encoder-0/w'][10,5:10]))
            summaries_merged_train = self._merge_summaries_scalars([loss, cost])
            
#            summaries = []
##            for value in [loss, cost]:
##                self.logger.debug("log scalar: %s" % value.op.name)
#                summaries.append(tf.summary.scalar(value.op.name, value))
#            self.logger.debug("log weights (histograms, images): %s" % self.model.w.keys())
#            self._append_summaries_hists(summaries, self.model.w.values())
#            self._append_summaries_var_imgs(summaries, self.model.w.values())
#            self._append_summaries_op_imgs(summaries, [self.model.p])
#            summaries_merged_val = tf.summary.merge(summaries)
            self._init_saver()
        
            fpath_save = os.path.join(dir_train, self._get_save_name())
            self.logger.debug("Save model at step %d to '%s'" % (itr_exp, fpath_save))
            self.saver.save(sess, fpath_save, global_step=itr_exp)   
            
            for epoch in xrange(self.training_epochs):
                self.logger.info("Start epoch %d, step %d" % (epoch, itr_exp))
                # Loop over all batches
                for itr_epoch in xrange(self.num_batches_train):
                    if self.tf_record_prefix is None:
                        batch_xs, _ = self.data.train.next_batch(self.batch_size_train)
                    else:
                        batch_xs, _, batch_os = sess.run([batch_xs_op, batch_ys_op, batch_os_op])
#                f = sess.run([augment_op], feed_dict={xx:batch_xs})
        #            batch_xs_as_img = tf.reshape(batch_xs, [-1, 28, 28, 1])
        #            rots_cur = np.random.choice(rotations, batch_size)
        #            batch_xs_as_img_rot = tf.contrib.image.rotate(batch_xs_as_img, rots_cur)
        #            # Run optimization op (backprop) and cost op (to get loss value)
        #            
        #            batch_xs_as_img, batch_xs_as_img_rot = \
        #                sess.run([batch_xs_as_img, batch_xs_as_img_rot],
        #                         feed_dict={ae1.x: batch_xs})
        #            f, a = plt.subplots(2, 10, figsize=(10, 2))
        #            for i in xrange(examples_to_show):
        #                print (batch_xs_as_img[i].shape, np.rad2deg(rots_cur)[i])
        #                a[0][i].imshow(np.squeeze(batch_xs_as_img[i]))
        #                a[1][i].imshow(np.squeeze(batch_xs_as_img_rot[i]))
        #            f.show()
        #            plt.draw()
        #            plt.waitforbuttonpress()
                    batch_xs_in = batch_xs
#                    print(len(sess.graph.get_operations()))
                    if self.do_augment_rot:
                        augment_op, _ = self.rotation_ops_multiset_train(3)
                        batch_xs_in = sess.run(augment_op, feed_dict={self.x: batch_xs})
#                        print(len(sess.graph.get_operations()))
#                    print(len(sess.graph.get_operations()))
#                    _, c, sess_summary = sess.run([optimizer, cost, summaries_merged_train],
#                                                  feed_dict={self.x: batch_xs})
                    fd = {self.x: batch_xs_in}
                    if self.do_reconstruct_original and dim_idx < 1:
                        fd[self.model.y_true] = batch_xs
                    _, c, nn, sess_summary = sess.run([optimizer, cost, self.model.enc_in, summaries_merged_train],
                                                      feed_dict=fd)
                    if self.is_time_to_track_train(itr_exp):
                        summary_writer_train.add_summary(sess_summary, itr_exp)
                    itr_exp += 1
                    itr_depth += 1
#                    import matplotlib.pyplot as plt
#                    f, a = plt.subplots(3, 10, figsize=(10, 2))
#                    for i in xrange(10):
#                        a[0][i].imshow(np.squeeze(batch_xs[i]).reshape(28,28))
#                        a[1][i].imshow(np.squeeze(batch_xs_in[i]).reshape(28,28))
#                        a[2][i].imshow(np.squeeze(nn[i]).reshape(28,28))
#                    f.show()
#                    plt.draw()
#                    plt.waitforbuttonpress()
                    # run metric op one more time, data in feed dict is dummy data, does not influence metric
#                if self.tf_record_prefix is None: # TODO extend to tfr
#                    _, sess_summary = sess.run([cost, summaries_merged_val],
#                                             feed_dict={self.x  : self.batch_viz_xs}
#                                             )
#                    _ = sess.run([cost],
#                                 feed_dict={self.x  : self.batch_viz_xs}
#                                )
#                    if self.is_time_to_track_val(itr_exp):
#                        summary_writer_val.add_summary(sess_summary, itr_exp)
                fpath_save = os.path.join(dir_train, self._get_save_name())
                self.logger.debug("Save model at step %d to '%s'" % (itr_exp, fpath_save))
                self.saver.save(sess, fpath_save, global_step=itr_exp)
                l = self.validate(sess, loss, dim_idx < 1)[0]
                self.logger.debug("validation loss after step %d: %f" % (itr_exp, l))
                result.last = l
                result.max = max(result.max, result.last)
        if self.tf_record_prefix is not None:
            coord.request_stop()
            coord.join(threads)
            self.logger.debug("Stop training queue")
        self.logger.info("Optimization Finished!")
        return result

    def validate(self, sess, loss, do_reconstruct_original=True):
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
                                                    num_threads=4
                                                    )
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for _ in xrange(num_batches_val):
            if self.tf_record_prefix is None:
                batch_xs, _ = self.data.validation.next_batch(self.batch_size_val,
                                                              shuffle=False)
            else:
                batch_xs, _, _ = sess.run([batch_xs_op, batch_ys_op, batch_os_op])
            batch_xs_in = batch_xs
            if self.do_augment_rot:
                augment_op, _ = self.rotation_ops_multiset_val(3)
                batch_xs_in = sess.run(augment_op, feed_dict={self.x: batch_xs})
            fd = {self.x: batch_xs_in}
            if do_reconstruct_original:
                fd[self.model.y_true] = batch_xs
            l = sess.run([loss], feed_dict=fd)
        if self.tf_record_prefix is not None:
            coord.request_stop()
            coord.join(threads)
            self.logger.debug("Stop validation queue")
        return l
            
    def _cost_loss(self, prefix):
        loss = self.model.cost(name=prefix + '/loss_reconstruction')
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
    
    def __init__(self, params):
        '''
        Constructor
        '''
        super(AERunner, self).__init__(params)
        self.data = self._init_data_mnist()
        self.num_batches_train = int(self.data.train.num_examples/self.batch_size_train)
        self.logger.debug("No. of batches per epoch: %d" % self.num_batches_train)
        self._check_validation_batch_size()
        self.stack_dims = params['stack_dims']
        self.logger.debug("Dims to stack: %s" % self.stack_dims)
        if self.tf_record_prefix is None:
            self.batch_viz_xs, self.batch_viz_ys = \
                self.data.validation.next_batch(self.batch_size_val,
                                                shuffle=False)
        self.do_reconstruct_original = params['do_reconstruct_original']
        self.logger.debug("Reconstruct original input: %s" % (['No', 'Yes'][self.do_reconstruct_original],))
        self._vars_added = []
        self.x = None