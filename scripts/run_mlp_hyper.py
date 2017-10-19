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
import pickle
import yaml
import numpy as np
from hyperopt import hp, fmin, tpe, space_eval, Trials
import transform.logging_utils as lu
from transform.cfg_utils import load_config
import run_mlp as script
logger = None

def objective(params):
    excl = ['base', 'run_dir', 'run_name', 'log_parent_dir']
    cfg = params['base']
    tmp = ''
    for key, value in params.items():
        if key not in excl:
            cfg[key] = value
            print(key, value)
            tmp += key + '-' + str(value)
    for key, value in cfg.items():
        if value is tuple():
            cfg[key] = []
    print(tmp)
    cfg['log_dir'] = os.path.join(params['log_parent_dir'], tmp)
    tmp = '_' + tmp
    fpath_cfg_dst = os.path.join(params['run_dir'], 'config%s.yml' % tmp)
    logger.debug("Write config %s" % (fpath_cfg_dst))
    with open(fpath_cfg_dst, 'w') as h:
        h.write(yaml.dump(cfg))
    ch_args_in = ['-c', fpath_cfg_dst,
                  '--log_dir', cfg['log_dir'],
                  '--run_name', params['run_name'] + tmp
                  ]
    args_ch = script.handleArgs(args=ch_args_in)
    result = \
        script.run(args_ch.run_name,
           args_ch.log_dir,
           args_ch.fpath_cfg_list
           )
    if result is None:
        raise(TypeError, "No result.")
    return result.min

def run(run_name, log_dir, fpath_cfg, nb_evals):
    run_dir = os.path.join(log_dir, run_name)
    run_dir_already_exists = False
    if not os.path.isdir(run_dir):
        os.makedirs(run_dir)
    else:
        run_dir_already_exists = True
    global logger
    logger = lu.setup_logging(os.path.join(run_dir, 'log.txt'))
    if run_dir_already_exists:
        logger.debug("Found run directory %s", run_dir)
    else:
        logger.debug("Created run directory %s", run_dir)
    fpath_trials = os.path.join(run_dir, "results.pkl")
    max_evals = nb_evals
    try:
        trials = pickle.load(open(fpath_trials, "rb"))
        logger.info("Loading saved trials from %s" % fpath_trials)
        max_evals += len(trials.trials)
        logger.debug("Rerunning from {} trials to add more.".format(
            len(trials.trials)))
    except:
        trials = Trials()
        logger.info("Starting trials from scratch.")
    
    cfg = load_config(fpath_cfg, logger)
    space = {
        'log_parent_dir' : os.path.abspath(os.path.expanduser(log_dir)),
        'run_dir' : run_dir,
        'run_name': run_name,
        'base'  : cfg,
        'learning_rate' : hp.loguniform('learning_rate', -7*np.log(10), -1*np.log(10))
        }
    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,
                trials=trials,
                max_evals=max_evals)
    print(best)
    print(space_eval(space, best))
    logger.info("Saave trials to %s" % fpath_trials)
    pickle.dump(trials, open(fpath_trials, "wb"))
    logger.info("Done.")

def handleArgs(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config",
                        dest="fpath_cfg", type=str, required=True,
                        help="Path to master config file")
    parser.add_argument("--log_dir", dest="log_dir", type=str,
                        default=os.path.abspath(os.path.curdir),
                        help="Set parent log directory for all runs")
    parser.add_argument("--run_name", dest="run_name", type=str,
                        default=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                        help="Set name for run")    
    parser.add_argument("--run_name_prefix", dest="run_name_prefix", type=str, default='',
                        help="Set prefix run name")
    parser.add_argument("--nb_evals", dest="nb_evals", type=int, required=True,
                        help="Maximum number of evaluations")
    
    return parser.parse_args(args=args)

if __name__ == '__main__':
    args = handleArgs()
    run(args.run_name_prefix + args.run_name,
        args.log_dir,
        args.fpath_cfg,
        args.nb_evals
        )
    pass
