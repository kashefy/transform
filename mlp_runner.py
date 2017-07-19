'''
Created on Jul 19, 2017

@author: kashefy
'''
import tensorflow as tf
#from tensorflow.python import debug as tf_debug
from abstract_runner import AbstractRunner, setup_optimizer

class MLPRunner(AbstractRunner):
    '''
    classdocs
    '''
    def learn(self, sess):
        summary_writer = tf.summary.FileWriter(self.run_dir,
                                               sess.graph)
        
        y_ = tf.placeholder("float", [None, self.model.n_outputs])
        cost = self.model.cost(y_, name="loss_classification")
        vars_new = self.model.vars_new()
        optimizer = setup_optimizer(cost, self.learning_rate, var_list=vars_new)
        vars_new = self.model.vars_new()
        self.init_vars(sess, vars_new)
        
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
                                              feed_dict={self.x: batch_xs,
                                                         y_: batch_ys})
                summary_writer.add_summary(sess_summary, itr_exp)
                itr_exp += 1
        self.logger.info("Classification Optimization Finished!")

    def __init__(self, params):
        '''
        Constructor
        '''
        super(MLPRunner, self).__init__(params)
        from tensorflow.examples.tutorials.mnist import input_data
        self.data = input_data.read_data_sets("MNIST_data", one_hot=True)
        self.num_barches = int(self.data.train.num_examples/self.batch_size)
        
        