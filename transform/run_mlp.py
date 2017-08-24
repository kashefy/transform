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
from datetime import datetime
import yaml
import tensorflow as tf
from mlp_runner import MLPRunner
from nideep.nets.mlp_tf import MLP
import logging_utils as lu
logger = None
from cfg_utils import load_config
  
def run(run_name, log_dir, fpath_cfg_list):
    run_dir = os.path.join(log_dir, run_name)
    os.makedirs(run_dir)
    global logger
    logger = lu.setup_logging(os.path.join(log_dir, run_name, 'log.txt'))
    logger.debug("Create run directory %s", run_dir)
    logger.info("Starting run %s" % run_name)
    
    cfg_list = []
    logger.debug("Got %d config files." % len(fpath_cfg_list))
    for cidx, fpath_cfg in enumerate(fpath_cfg_list):
        logger.debug("Loading config from %s" % fpath_cfg)
        cfg = load_config(fpath_cfg)
        cfg['log_dir'] = os.path.expanduser(log_dir)
        cfg['run_name'] = run_name
        fname_cfg = os.path.basename(fpath_cfg)
        fpath_cfg_dst = os.path.join(run_dir, 'config_%d.yml' % cidx)
        logger.debug("Write config %s to %s" % (fname_cfg, fpath_cfg_dst))
        with open(fpath_cfg_dst, 'w') as h:
            h.write(yaml.dump(cfg))
        cfg_list.append(cfg)
    
    cfg = cfg_list[0]
    mlp_runner = MLPRunner(cfg)
    n_input = mlp_runner.data.train.images.shape[-1]

    # Launch the graph
    with tf.Session() as sess:
        n_classes = mlp_runner.data.train.labels.shape[-1]
        classifier_params = {
            'n_nodes'   : [n_classes],
            'n_input'   : n_input,
            'prefix'    : cfg['prefix'],
            }
        net = MLP(classifier_params)
        net.x = tf.placeholder("float", [None, n_input])
        net.build()
        mlp_runner.x = net.x
        mlp_runner.model = net
        mlp_runner.learn(sess)
    logger.info("Finished run %s" % run_name)
    lu.close_logging(logger)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", action='append',
                        dest="fpath_cfg_list", type=str, required=True,
                        help="Paths to config files")
    parser.add_argument("--log_dir", dest="log_dir", type=str,
                        default='/home/kashefy/models/ae/log_simple_stats',
                        help="Set parent log directory for all runs")
    parser.add_argument("--run_name", dest="run_name", type=str,
                        default=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                        help="Set name for run")    
    parser.add_argument("--run_name_prefix", dest="run_name_prefix", type=str, default='',
                        help="Set prefix run name")
    args = parser.parse_args()
    run(args.run_name_prefix + args.run_name,
        args.log_dir,
        args.fpath_cfg_list)
    
    pass
