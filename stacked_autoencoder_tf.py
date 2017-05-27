'''
Created on May 26, 2017

@author: kashefy
'''
import tensorflow as tf
from autoencoder_tf import Autoencoder as AE

class StackedAutoencoder(object):
    '''
    classdocs
    '''
    def _in_op_cur(self):
        
        in_op = None
        if len(self.sae) > 0:
            in_op = self.sae[-1].representation()
        else:
            in_op = self.in_op
        return in_op
    
    def stack(self, dim):
        
        self.dims.append(dim)
        in_op = self._in_op_cur()
        # Network Parameters
        ae_params = {
            'n_hidden_1':  dim,
            'n_input'  :   int(in_op.get_shape()[-1]),
            'depth'    :   len(self.sae)+1
             }
        ae = AE(ae_params)
        ae.x = in_op
        _, _ = ae.construct_model()
        
        if len(self.sae) > 0:
            self.sae[-1].decoder(ae.y_pred)
        self.sae.append(ae)
            
        # Targets (Labels) are the input data.
        self._y_true = self.sae[0].x
        
    def y_true(self):
        return self._y_true
            
    def cost(self):
        return self.sae[0].cost_cross_entropy(self._y_true)
        
    def vars_new(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                 scope=self.sae[-1].name_scope)
    
    @property
    def x(self):
        return self.sae[0].x        
    
    @property
    def y_pred(self):
        return self.sae[0].y_pred

    def __init__(self, params):
        '''
        Constructor
        '''
        self.dims = []
        self.in_op = params['in_op']
        self.sae = []
        