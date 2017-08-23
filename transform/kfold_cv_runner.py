'''
Created on Jul 31, 2017

@author: kashefy
'''
from abc import ABCMeta, abstractmethod
from sklearn.model_selection import KFold
from abstract_runner import AbstractRunner

#X = ["a", "a", "b", "c", "c", "c"]
#k_fold = KFold(n_splits=3)
#

class KFoldCVRunner(AbstractRunner):
    '''
    classdocs
    '''
    def learn(self, sess):
        #for train_indices, val_indices in self.k_fold.split(X):
        for k in range(self.num_folds):
            self._learn_fold(k)
        
    @abstractmethod
    def _learn_fold(self, sess, k):
        pass
    
    def __init__(self, params):
        '''
        Constructor
        '''
        super(KFoldCVRunner, self).__init__(params)
        self.num_folds = params.num_folds
        self.logger.debug("No. of CV folds: %s", self.num_folds)
        self.k_fold = KFold(n_splits=self.num_folds)