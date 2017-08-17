'''
Created on Jul 19, 2017

@author: kashefy
'''
import os
import tensorflow as tf
#from tensorflow.python import debug as tf_debug
from abstract_runner import AbstractRunner, setup_optimizer
from nideep.datasets.mnist.mnist_tf import MNIST

class AERunner(AbstractRunner):
    '''
    classdocs
    '''
    def learn(self, sess):
        dir_train = os.path.join(self.run_dir, self.prefix,  'train')
        summary_writer_train = tf.summary.FileWriter(dir_train,
                                                     sess.graph)
        itr_exp = 0
        for dim in self.stack_dims:
            itr_depth = 0
            self.logger.debug('Stacking %d nodes.' % dim)
            self.model.stack(dim)
            cost = self.model.cost(name='train/loss_reconstruction')
            vars_new = self.model.vars_new()
            optimizer = setup_optimizer(cost, self.learning_rate, var_list=vars_new)
            vars_new = self.model.vars_new()
            self.init_vars(sess, vars_new)
           
            self.logger.debug('encoder-0: %s' % sess.run(self.model.sae[0].w['encoder-0/w'][10,5:10]))
            summaries_merged_train = self._merge_summaries([cost])
            
            self._init_saver()
            
            for epoch in xrange(self.training_epochs):
                self.logger.info("Epoch %d, step %d" % (epoch, itr_exp))
                # Loop over all batches
                for itr_epoch in xrange(5):#self.num_batches):
                    batch_xs, _ = self.data.train.next_batch(self.batch_size)
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
                    _, c, sess_summary = sess.run([optimizer, cost, summaries_merged_train],
                                                  feed_dict={self.model.x: batch_xs})
                    summary_writer_train.add_summary(sess_summary, itr_exp)
                    itr_exp += 1
                    itr_depth += 1
                fpath_save = os.path.join(dir_train, self._get_save_name())
                self.logger.debug("Save model at step %d to '%s'" % (itr_exp, fpath_save))
                self.saver.save(sess, fpath_save, global_step=itr_exp)               
            self.logger.info("Optimization Finished!")
            self.logger.debug('encoder-0: %s' % sess.run(self.model.sae[0].w['encoder-0/w'][10,5:10]))
#        if dim == 128:
#            print('encoder_2',sess.run(sae.sae[1].w['encoder_2/w'][10,5:10]))
#            encode_decode = sess.run(
#                sae.p, feed_dict={sae.x: mnist.test.images[:args.examples_to_show]})
        # Compare original images with their reconstructions
#    fig, a = plt.subplots(2, 10, figsize=(10, 2))
#    for i in xrange(args.examples_to_show):
#        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)), clim=(0.0, 1.0))
#        a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)), clim=(0.0, 1.0))
#    fig.savefig(os.path.join(args.log_dir, run_dir, 'train_layerwise_reconstruct.png'))
        
    def __init__(self, params):
        '''
        Constructor
        '''
        super(AERunner, self).__init__(params)
        self.data = MNIST.read_data_sets("MNIST_data",
                                         validation_size=self.validation_size)
        self.num_batches = int(self.data.train.num_examples/self.batch_size)
        self.logger.debug("No. of batches per epoch: %s", self.num_batches)
        self.stack_dims = params['stack_dims']
        self.logger.debug("Stack dims: %s", self.stack_dims)
        self.prefix = 'reconstruction'
        