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
import shutil
from datetime import datetime
import yaml
import tensorflow as tf
from transform.ae_runner import AETRFRunner
from transform.mlp_runner import MLPRunner
from transform.autoencoder_tf import Autoencoder as AE
from nideep.nets.mlp_tf import MLP
import transform.logging_utils as lu
from transform.cfg_utils import load_config
logger = None

def run(run_name, log_dir, fpath_cfg_list,
        fpath_meta, dir_checkpoints):
    run_dir = os.path.join(log_dir, run_name)
    if os.path.isdir(run_dir):
        shutil.rmtree(run_dir)
    os.makedirs(run_dir)
    global logger
    logger = lu.setup_logging(os.path.join(log_dir, run_name, 'log.txt'))
    logger.debug("Create run directory %s", run_dir)
    logger.info("Starting run %s" % run_name)
    # -90 (cw) to 90 deg (ccw) rotations in 15-deg increments
    #rotations = np.deg2rad(np.linspace(-90, 90, 180/(12+1), endpoint=True)).tolist()
    
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
    
    reuse = fpath_meta is not None and dir_checkpoints is not None
    if reuse:
        reuse = True
        trained_model = tf.train.import_meta_graph(fpath_meta)
        
    cfg = cfg_list[0]
    ae_runner = AETRFRunner(cfg)
    n_input = ae_runner.data.train.images.shape[-1]
    ae_params = {
        'n_nodes'   :  [256],
        'n_input'   :  int(n_input),
        'prefix'    :  cfg['prefix'],
        'reuse'     :  reuse,
         }
    ae_runner.model = AE(ae_params)
    ae_runner.model.x = tf.placeholder("float", [None, n_input])
    ae_runner.model.build()
    
    cfg = cfg_list[1]
    mlp_runner = MLPRunner(cfg)
    
    # Launch the graph
    with tf.Session() as sess:
        if fpath_meta is not None and dir_checkpoints is not None:
            trained_model.restore(sess, tf.train.latest_checkpoint(dir_checkpoints))
        ae_runner.learn(sess)
        
        n_classes = mlp_runner.data.train.labels.shape[-1]
        classifier_params = {
            'n_nodes'   : [n_classes],
            'n_input'   : ae_runner.model.representation.get_shape()[-1].value,
            'prefix'    : cfg['prefix'],
            }
        net = MLP(classifier_params)
        net.x = ae_runner.model.representation
        net.build()
        mlp_runner.x = ae_runner.model.x
        mlp_runner.model = net
        mlp_runner.learn(sess)
    logger.info("Finished run %s" % run_name)
    lu.close_logging(logger)
    
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
    parser.add_argument("--fpath_meta", dest='fpath_meta', type=str, default=None,
                        help="Path to file with meta graph for restoring trained models.")
    parser.add_argument("--dir_checkpoints", dest='dir_checkpoints', type=str, default=None,
                        help="Checkpoint directory for restoring trained models.")
    return parser.parse_args(args=args)

if __name__ == '__main__':

    args = handleArgs()
    run(args.run_name_prefix + args.run_name,
        args.log_dir,
        args.fpath_cfg_list,
        args.fpath_meta, args.dir_checkpoints,
        )
    
    pass