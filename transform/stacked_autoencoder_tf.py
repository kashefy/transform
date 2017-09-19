'''
Created on May 26, 2017

@author: kashefy
'''
from nideep.nets.abstract_net_tf import AbstractNetTF
from autoencoder_tf import Autoencoder as AE

class StackedAutoencoder(AbstractNetTF):
    '''
    classdocs
    '''
    def _init_learning_params_scoped(self):
        pass
    
    def _init_ops(self):
        pass
    
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
            'n_nodes'   :  [dim],
            'n_input'   :  int(in_op.get_shape()[-1]),
            'prefix'    :  '%s-%d' % (self.prefix, len(self.sae)+1),
            'reuse'     :  self.reuse,
             }
        ae = AE(ae_params)
        ae.x = in_op
        _, _ = ae.build()
        
        if len(self.sae) > 0:
            self.sae[-1].decoder(ae.p)
        self.sae.append(ae)
            
        # Targets (Labels) are the input data.
        self._y_true = self.sae[-1].x
        
    def y_true(self):
        return self._y_true
            
    def cost(self, name=None):
        return self.sae[-1].cost_cross_entropy(self._y_true, name=name)
        
    def vars_new(self):
        return self.sae[-1].vars_new()
    
    @property
    def representation(self):
        return self.sae[-1].representation()
    
    @property
    def x(self):
        return self.sae[0].x        
    
    @property
    def p(self):
        return self.sae[0].p

    @property
    def logits(self):
        return self.sae[0].logits

    @property
    def w(self):
        w = {}
        for ae in self.sae:
            for k in ae.w:
                w[k] = ae.w[k]
        return w

    @property
    def vars_restored(self):
        self._vars_restored = []
        for ae in self.sae:
            self._vars_restored.extend(ae.vars_restored)
        return self._vars_restored

    def __init__(self, params):
        '''
        Constructor
        '''
        self.dims = []
        self.in_op = params['in_op']
        params['n_input'] = int(self.in_op.get_shape()[-1])
        super(StackedAutoencoder, self).__init__(params)
        self.sae = []
        