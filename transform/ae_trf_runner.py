'''
Created on Jul 19, 2017

@author: kashefy
'''
import os
import numpy as np
import tensorflow as tf
#from tensorflow.python import debug as tf_debug
from abstract_runner import setup_optimizer
from transform.ae_runner import AERunner
from transform.autoencoder_tf import Autoencoder as AE
from nideep.datasets.mnist.mnist_tf import MNIST

class AETRFRunner(AERunner):
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
        
        ae_params = {
        'n_nodes'   :  self.model.n_nodes,
        'n_input'   :  self.data.train.images.shape[-1],
        'prefix'    :  self.model.prefix,
        'reuse'     :  True,
         }
        
        # -90 (cw) to 90 deg (ccw) rotations in 15-deg increments
        rotations = np.deg2rad(np.linspace(-90, 90, 180/(12+1), endpoint=True)).tolist()
        print np.rad2deg(rotations)

            
        aerot = AE(ae_params)
        aerot.x = tf.placeholder("float", [None, self.model.n_input])
        aerot.build()
        aerot.p = aerot.representation()
        cost_constraint_op = aerot.cost_euclidean(self.model.representation(),
                                                  name=self.dirname('train') + '/loss_constraint')
        
        cost, loss = self._cost_loss(self.dirname('train'))
        vars_new = self.model.vars_new()
        self.logger.debug('Variables added: %s' % [v.name for v in vars_new])
        self._vars_added.append(vars_new)
        optimizer = setup_optimizer(cost, self.learning_rate, var_list=vars_new)
        vars_new = self.model.vars_new()
#            self.logger.debug('encoder-0: %s' % sess.run(self.model.sae[0].w['encoder-0/w'][10,5:10]))
        

        
        self.init_vars(sess, vars_new)
        self.logger.debug('encoder-0: %s' % sess.run(self.model.w['encoder-0/w'][10,5:10]))
        summaries_merged_train = self._merge_summaries_scalars([loss, cost])
        
        summaries = []
#            for value in [loss, cost]:
#                self.logger.debug("log scalar: %s" % value.op.name)
#                summaries.append(tf.summary.scalar(value.op.name, value))
        self.logger.debug("log weights (histograms, images): %s" % self.model.w.keys())
        self._append_summaries_hists(summaries, self.model.w.values())
        self._append_summaries_var_imgs(summaries, self.model.w.values())
        self._append_summaries_op_imgs(summaries, [self.model.p])
        summaries_merged_val = tf.summary.merge(summaries)
        
        self._init_saver()
    
        fpath_save = os.path.join(dir_train, self._get_save_name())
        self.logger.debug("Save model at step %d to '%s'" % (itr_exp, fpath_save))
        self.saver.save(sess, fpath_save, global_step=itr_exp)   
        
        for epoch in xrange(self.training_epochs):
            self.logger.info("Start epoch %d, step %d" % (epoch, itr_exp))
            # Loop over all batches
            for itr_epoch in xrange(self.num_batches_train):
                batch_xs, _ = self.data.train.next_batch(self.batch_size_train)
#                
#                batch_xs_as_img_op = tf.reshape(self.model.x, [-1, 28, 28, 1])
#                rots_cur = np.random.choice(rotations, self.batch_size_train)
#                rot_op = tf.contrib.image.rotate(batch_xs_as_img_op, rots_cur)
#                flatten_op = tf.reshape(rot_op, [-1, self.data.train.images.shape[-1]])
#
#                flattened, _, c, sess_summary = \
#                    sess.run([flatten_op,
#                              optimizer, cost, summaries_merged_train],
#                             feed_dict={self.model.x: batch_xs})
                _, c, sess_summary = \
                    sess.run([
                              optimizer, cost, summaries_merged_train],
                             feed_dict={self.model.x: batch_xs})
#                cost_constraint = sess.run([cost_constraint_op],
#                                           feed_dict={self.model.x: batch_xs,
#                                             aerot.x: flattened}
#                                           )
                print itr_exp
#                print cost_constraint, r0.flatten()[128:132], p0.flatten()[128:132]
#                import matplotlib.pyplot as plt
#                f, a = plt.subplots(2, 10, figsize=(10, 2))
#                for i in xrange(10):
##                    print (batch_xs_as_img[i].shape, np.rad2deg(rots_cur)[i])
#                    a[0][i].imshow(np.squeeze(batch_xs_as_img[i]))
#                    a[1][i].imshow(np.squeeze(batch_xs_as_img_rot[i]))
#                f.show()
#                plt.draw()
#                plt.waitforbuttonpress()
                if self.is_time_to_track_train(itr_exp):
                    summary_writer_train.add_summary(sess_summary, itr_exp)
                itr_exp += 1
                # run metric op one more time, data in feed dict is dummy data, does not influence metric
            _, sess_summary = sess.run([cost, summaries_merged_val],
                                     feed_dict={self.model.x  : self.batch_viz_xs}
                                     )
            if self.is_time_to_track_val(itr_exp):
                summary_writer_val.add_summary(sess_summary, itr_exp)
            fpath_save = os.path.join(dir_train, self._get_save_name())
            self.logger.debug("Save model at step %d to '%s'" % (itr_exp, fpath_save))
            self.saver.save(sess, fpath_save, global_step=itr_exp)               
        self.logger.info("Optimization Finished!")
        self.logger.debug('encoder-0: %s' % sess.run(self.model.w['encoder-0/w'][10,5:10]))

    def validate(self, sess):
        pass
    
    def _cost_loss(self, prefix):
        loss = self.model.cost_cross_entropy(self.model.x, name=prefix + '/loss_reconstruction')
        if self.lambda_l2 != 0:
            regularization = self._regularization(name=prefix + '/regularization_l2')
            cost = tf.add(loss, self.lambda_l2 * regularization,
                          name=prefix + '/cost')
        else:
            cost = loss
        return cost, loss
    
    def __init__(self, params):
        '''
        Constructor
        '''
        super(AETRFRunner, self).__init__(params)