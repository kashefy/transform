'''
Created on Jul 19, 2017

@author: kashefy
'''
import tensorflow as tf
#from tensorflow.python import debug as tf_debug
from abstract_runner import AbstractRunner, setup_optimizer
from nideep.nets.mlp_tf import MLP

class MLPRunner(AbstractRunner):
    '''
    classdocs
    '''
    def learn(self, sess):
        summary_writer = tf.summary.FileWriter(self.run_dir,
                                               sess.graph)
        
        y_ = tf.placeholder("float", [None, self.net.n_outputs])
        cost = self.net.cost(y_, name="loss_classification")
        vars_new = self.net.vars_new()
        optimizer = setup_optimizer(cost, self.learning_rate, var_list=vars_new)
        vars_new = self.net.vars_new()
        self.init_vars(sess, vars_new)
                
        # Training cycle
        print('encoder-1', sess.run(sae.sae[0].w['encoder-1/w'][10,5:10]))
        
        for value in [cost]:
            print("log scalar", value.op.name)
            tf.summary.scalar(value.op.name, value)
        
        summaries = tf.summary.merge_all()
            
        itr_exp = 0
        for epoch in xrange(self.training_epochs):
            # Loop over all batches
            for itr_epoch in xrange(self.num_barches):
                batch_xs, batch_ys = self.data.train.next_batch(self.batch_size)
                _, c, sess_summary = sess.run([optimizer, cost, summaries],
                                              feed_dict={sae.x: batch_xs,
                                                         y_: batch_ys})
                summary_writer.add_summary(sess_summary, itr_exp)
                itr_exp += 1
        self.logger.info("Classification Optimization Finished!")
        self.logger.debug('encoder-1: %s' % sess.run(sae.sae[0].w['encoder-1/w'][10,5:10]))
    #        if dim == 128:
    #            print('encoder_2',sess.run(sae.sae[1].w['encoder_2/w'][10,5:10]))

    def __init__(self, params):
        '''
        Constructor
        '''
        super(MLPRunner, self).__init__(params)
        from tensorflow.examples.tutorials.mnist import input_data
        self.data = input_data.read_data_sets("MNIST_data", one_hot=True)
        self.num_barches = int(self.data.train.num_examples/self.batch_size)
        n_classes = self.data.test.labels.shape[-1]
        classifier_params = {
            'n_outputs': n_classes,
            'n_input': ,
            'prefix': 'mlp_',
            }
        self.net = MLP(classifier_params)
        