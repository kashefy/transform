# -*- coding: utf-8 -*-

""" Auto Encoder Example.
Using an auto encoder on MNIST handwritten digits.
References:
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.
Links:
    [MNIST Dataset] http://yann.lecun.com/exdb/mnist/
"""
from __future__ import division, print_function, absolute_import

import argparse
import os
import logging
from datetime import datetime
import numpy as np
import yaml
import tensorflow as tf
from mlp_runner import MLPRunner
from nideep.nets.mlp_tf import MLP
logger = None

def setup_logging(fpath,
                  name=None, 
                  level=logging.DEBUG):
    if name is None:
        logger = logging.getLogger()
    else:
        logger = logging.getLogger(name)
    logger.setLevel(level)
    fh = logging.FileHandler(fpath)
    ch = logging.StreamHandler()
    
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    for h in [fh, ch]:
        fh.setLevel(level)
        h.setFormatter(formatter)
        # add the handlers to the logger
        logger.addHandler(h)
    return logger
    
def close_logging(logger):
    # remember to close the handlers
    for handler in logger.handlers:
        handler.close()
        logger.removeFilter(handler)
        
def load_config(fpath):
    _, ext = os.path.splitext(fpath)
    if not (ext.endswith('yml') or ext.endswith('yaml')):
        logger.warning("Config file does not appear to be a yaml file.")
    fpath = os.path.expanduser(fpath)
    with open(fpath, 'r') as h:
        cfg = yaml.load(h)
    # set defaults if not already set
    default_items = {'learning_rate'    : 0.5,
                     'training_epochs'  : 2, # no. of epochs per stage
                     'batch_size'       : 64,
                     "num_folds"        : 3,
                     "prefix"           : '',
                     "lambda_l2"        : 0,
                     "logger_name"      : logger.name,
                     }
    for k in default_items.keys():
        cfg[k] = cfg.get(k, default_items[k])
    return cfg
  
def run_mlp(args):
    args.run_name = args.run_name_prefix + args.run_name
    run_dir = os.path.join(args.log_dir, args.run_name)
    os.makedirs(run_dir)
    global logger
    logger = setup_logging(os.path.join(args.log_dir, args.run_name, 'log.txt'))
    logger.debug("Create run directory %s", run_dir)
    logger.info("Starting run %s" % args.run_name)
    # -90 (cw) to 90 deg (ccw) rotations in 15-deg increments
    #rotations = np.deg2rad(np.linspace(-90, 90, 180/(12+1), endpoint=True)).tolist()
    
    logger.debug("Loading config from %s" % args.fpath_cfg)
    cfg = load_config(args.fpath_cfg)
    cfg['log_dir'] = os.path.expanduser(args.log_dir)
    cfg['run_name'] = args.run_name
    fname_cfg = 'config.yaml' #os.path.basename(params.fpath_cfg)
    with open(os.path.join(run_dir, fname_cfg), 'w') as h:
        h.write(yaml.dump(cfg))
    
    mlp_runner = MLPRunner(cfg)
    n_input = mlp_runner.data.train.images.shape[-1]

    # Launch the graph
    with tf.Session() as sess:
        n_classes = mlp_runner.data.train.labels.shape[-1]
        classifier_params = {
            'n_nodes': [n_classes],
            'n_input': n_input,
            'prefix': 'mlp',
            }
        net = MLP(classifier_params)
        net.x = tf.placeholder("float", [None, n_input])
        net.build()
        mlp_runner.x = net.x
        mlp_runner.model = net
        mlp_runner.learn(sess)
    logger.info("Finished run %s" % args.run_name)
    close_logging(logger)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", dest="fpath_cfg", type=str,
                        help="Path to config file")
    parser.add_argument("--log_dir", dest="log_dir", type=str, default='/home/kashefy/models/ae/log_simple_stats',
                        help="Set parent log directory for all runs")
    parser.add_argument("--run_name", dest="run_name", type=str, default=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                        help="Set name for run")    
    parser.add_argument("--run_name_prefix", dest="run_name_prefix", type=str, default='',
                        help="Set prefix run name")    
    
    args = parser.parse_args()
    run_mlp(args)
    
    pass
