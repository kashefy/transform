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
from transform.mlp_runner import MLPRunner, augment_rotation
from nideep.nets.mlp_tf import MLP
import transform.logging_utils as lu
from transform.cfg_utils import load_config
logger = None
  
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
        cfg = load_config(fpath_cfg, logger)
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
    result = None
    grph = tf.Graph()
    with grph.as_default() as g:
        with tf.Session(graph=g) as sess:
            n_classes = mlp_runner.data.train.labels.shape[-1]
            cfg['n_nodes'].append(n_classes)
            classifier_params = {
                'n_nodes'   : cfg['n_nodes'],
                'n_input'   : n_input,
                'prefix'    : cfg['prefix'],
                }
            net = MLP(classifier_params)
            in_ = tf.placeholder("float", [None, n_input])
#            net.x = augment_rotation(in_,
#                                            -90, 90, 15,
#                                            cfg['batch_size_train'])
            net.x = in_
            net.build()
            mlp_runner.x = in_
            mlp_runner.model = net
            result = mlp_runner.learn(sess)
        logger.info("Finished run %s" % run_name)
    lu.close_logging(logger)
    return result
    
def handleArgs(args=None):
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
    return parser.parse_args(args=args)

if __name__ == '__main__':
    args = handleArgs()
    run(args.run_name_prefix + args.run_name,
        args.log_dir,
        args.fpath_cfg_list
        )
    pass
