'''
Created on Aug 24, 2017

@author: kashefy
'''
import os
import yaml

def load_config(fpath, logger):
    _, ext = os.path.splitext(fpath)
    if not (ext.endswith('yml') or ext.endswith('yaml')):
        logger.warning("Config file does not appear to be a yaml file.")
    fpath = os.path.expanduser(fpath)
    with open(fpath, 'r') as h:
        cfg = yaml.load(h)
    # set defaults if not already set
    default_items = {'learning_rate'    : 0.1,
                     'training_epochs'  : 2, # no. of epochs per stage
                     'batch_size_train' : 16,
                     'batch_size_val'   : 16,
                     "num_folds"        : 3,
                     "prefix"           : os.path.splitext(os.path.basename(fpath))[0],
                     "lambda_l2"        : 1.0,
                     "logger_name"      : logger.name,
                     "n_nodes"          : [],
                     "track_interval_train" : 1,
                     "track_interval_val"   : 1,
                     }
    for k in default_items.keys():
        cfg[k] = cfg.get(k, default_items[k])
    return cfg