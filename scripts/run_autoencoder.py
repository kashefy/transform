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
from transform.ae_runner import AERunner
#from transform.mlp_runner import MLPRunner as MLPRunner
from transform.mlp2task_runner import MLP2TaskRunner as MLPRunner
from transform.stacked_autoencoder_tf import StackedAutoencoder as SAE
from nideep.nets.mlp_tf import MLP
import transform.logging_utils as lu
from transform.cfg_utils import load_config
from transform.augmentation import rotation_rad
logger = None

def run(run_name, args):
    if args.run_dir is None:
        run_dir = os.path.join(args.log_dir, run_name)
    else:
        run_dir = args.run_dir
    run_dir_already_exists = False
    if not os.path.isdir(run_dir):
        os.makedirs(run_dir)
    else:
        run_dir_already_exists = True
    global logger
    logger = lu.setup_logging(os.path.join(args.log_dir, 'log.txt'),
                              name=[args.logger_name, None][args.logger_name_none])
    if run_dir_already_exists:
        logger.debug("Found run directory %s", run_dir)
    else:
        logger.debug("Created run directory %s", run_dir)
    logger.info("Starting run %s" % run_name)
    cfg_list = []
    logger.debug("Got %d config files." % len(args.fpath_cfg_list))
    for cidx, fpath_cfg in enumerate(args.fpath_cfg_list):
        logger.debug("Loading config from %s" % fpath_cfg)
        cfg = load_config(fpath_cfg, logger)
        cfg['log_dir'] = os.path.expanduser(args.log_dir)
        cfg['run_name'] = run_name
        cfg['run_dir'] = os.path.expanduser(run_dir)
        fpath_cfg_dst = os.path.join(run_dir, 'config_%d.yml' % cidx)
        logger.debug("Write config %s to %s" % (os.path.basename(fpath_cfg),
                                                fpath_cfg_dst))
        with open(fpath_cfg_dst, 'w') as h:
            h.write(yaml.dump(cfg))
        cfg_list.append(cfg)
    reuse = args.fpath_meta is not None and args.dir_checkpoints is not None
    if reuse:
        trained_model = tf.train.import_meta_graph(args.fpath_meta)
    cfg = cfg_list[0]
    ae_runner = AERunner(cfg)
    n_input = reduce(lambda x, y: x * y, ae_runner.data.train.images.shape[1:], 1)
    config = tf.ConfigProto()
    logger.debug('per_process_gpu_memory_fraction set to %f' % args.per_process_gpu_memory_fraction)
    config.gpu_options.per_process_gpu_memory_fraction = args.per_process_gpu_memory_fraction
    grph = tf.Graph()
    with grph.as_default() as g:
        in_op = tf.placeholder("float", [None, n_input])
        sae_params = {
                    'in_op'     : in_op,
                    'reuse'     : reuse,
                    'prefix'    : cfg['prefix'],
                    'do_denoising'      : cfg['do_denoising'],
                    'input_noise_std'   : cfg['input_noise_std'],
                    }
        ae_runner.model = SAE(sae_params)
        cfg = cfg_list[1]
        mlp_runner = MLPRunner(cfg)
    
        # Launch the graph
        result_mlp = None
        result_mlp_orient = None
        result_mlp_fine = None
        result_mlp_fine_orient = None
        tasks = []
        if mlp_runner.do_task_recognition:
            tasks.append('recognition')
        if mlp_runner.do_task_orientation:
            tasks.append('orientation')
        with tf.Session(graph=g, config=config) as sess:
            if args.fpath_meta is not None and args.dir_checkpoints is not None:
                trained_model.restore(sess, tf.train.latest_checkpoint(args.dir_checkpoints))
            result_ae = ae_runner.learn(sess)
            n_classes = mlp_runner.data.train.labels.shape[-1]
            classifier_params = {
                'n_nodes'   : [n_classes],
                'n_input'   : ae_runner.model.representation.get_shape()[-1].value,
                'prefix'    : cfg['prefix'],
                'branch'        : cfg.get('branch', len(cfg['n_nodes'])), # substract additional because of decision layer
                'logger_name'   : cfg['logger_name'],
                }
            net = MLP(classifier_params)
            net.x = ae_runner.model.representation
            net.build()
            mlp_runner.x = sae_params['in_op']
            mlp_runner.orient_ = tf.placeholder("float", shape=[None, len(rotation_rad(-60,60,15))])
            mlp_runner.model = net
            result_mlp, result_mlp_orient = mlp_runner.learn(sess)
#            logger.debug('encoder-0: %s' % sess.run(ae_runner.model.sae[0].w['encoder-0/w'][10,5:10]))
            mlp_runner.do_finetune = True
#            result_mlp_fine = mlp_runner.learn(sess)
            result_mlp_fine, result_mlp_fine_orient = mlp_runner.learn(sess)
#            logger.debug('encoder-0: %s' % sess.run(ae_runner.model.sae[0].w['encoder-0/w'][10,5:10]))
    logger.info("Finished run %s" % run_name)
    lu.close_logging(logger)
    return result_ae,\
        result_mlp, result_mlp_orient,\
        result_mlp_fine, result_mlp_fine_orient, tasks
    
def handleArgs(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", action='append',
                        dest="fpath_cfg_list", type=str, required=True,
                        help="Paths to config files")
    parser.add_argument("--log_dir", dest="log_dir", type=str,
                        default='/home/kashefy/models/ae/log_simple_stats',
                        help="Set parent log directory for all runs")
    parser.add_argument("--logger_name", dest="logger_name", type=str,
                        default=__name__,
                        help="Set name for process logging")
    parser.add_argument('--logger_name_none', action='store_true')
    parser.add_argument("--run_name", dest="run_name", type=str,
                        default=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                        help="Set name for run")    
    parser.add_argument("--run_name_prefix", dest="run_name_prefix", type=str, default='',
                        help="Set prefix run name")
    parser.add_argument("--fpath_meta", dest='fpath_meta', type=str, default=None,
                        help="Path to file with meta graph for restoring trained models.")
    parser.add_argument("--dir_checkpoints", dest='dir_checkpoints', type=str, default=None,
                        help="Checkpoint directory for restoring trained models.")
    parser.add_argument("--run_dir", dest="run_dir", type=str, default=None,
                        help="Set run directory")
    parser.add_argument("--per_process_gpu_memory_fraction", dest="per_process_gpu_memory_fraction",
                        type=float, default=1.,
                        help="Tensorflow's gpu option per_process_gpu_memory_fraction")
    parser.add_argument("--data_dir", dest="data_dir", type=str,
                        required=True,
                        help="Path to data directory")
    parser.add_argument("--tf_record_prefix", dest="tf_record_prefix", type=str,
                        help="filename prefix for tf records files")
    parser.add_argument("--data_seed", dest="data_seed", type=int,
                        default=None,
                        help="seed for data generation")
    return parser.parse_args(args=args)

if __name__ == '__main__':

    args = handleArgs()
    run(args.run_name_prefix + args.run_name,
        args,
        )