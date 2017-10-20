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
import sys
from datetime import datetime
import pickle
import json
import yaml
import numpy as np
from hyperopt import hp, fmin, tpe, space_eval, Trials
import transform.logging_utils as lu
from transform.cfg_utils import load_config
import run_mlp as script
logger = None

def get_items_suffix(items):
    return '_'.join(['-'.join([k, str(v)]) for k, v in items])

def update_cfg(base, params):
    excl = ['base', 'run_dir', 'run_name', 'log_dir']
    items_new = []
    for key, value in params.items():
        if key not in excl:
            base[key] = value
            items_new.append([key, value])
    for key, value in base.items():
        if isinstance(value, tuple):
            base[key] = list(value)
    suffix = get_items_suffix(items_new)
    base['log_dir'] = os.path.join(params['run_dir'], suffix)
    base['run_dir'] = os.path.join(params['run_dir'], suffix)
    return base, items_new, suffix

def objective(params):
    cfg, _, suffix = update_cfg(params['base'], params)
    fpath_cfg_dst = os.path.join(params['run_dir'], 'config_%s.yml' % suffix)
    logger.debug("Write config %s" % (fpath_cfg_dst))
    with open(fpath_cfg_dst, 'w') as h:
        h.write(yaml.dump(cfg))
    ch_args_in = ['-c', fpath_cfg_dst,
                  '--log_dir', cfg['log_dir'],
                  '--run_name', params['run_name'] + '_' + suffix
                  ]
    args_ch = script.handleArgs(args=ch_args_in)
    result = \
        script.run(args_ch.run_name,
           args_ch
           )
    if result is None:
        raise(TypeError, "No result.")
    return result.min

if sys.version_info.major < 3:  # Python 2?
    # Using exec avoids a SyntaxError in Python 3.
    exec("""def reraise(exc_type, exc_value, exc_traceback=None):
                raise exc_type, exc_value, exc_traceback""")
else:
    def reraise(exc_type, exc_value, exc_traceback=None):
        if exc_value is None:
            exc_value = exc_type()
        if exc_value.__traceback__ is not exc_traceback:
            raise exc_value.with_traceback(exc_traceback)
        raise exc_value
    
def add_space(space_base, space_dir):
    sys.path.append(args.space_dir)
    try:
        import space
        fpath_space = space.__file__
        from space import space as space_add
    except ImportError as e:
        module_name = str(e).split(' ')[-1]
        expected_module_path = os.path.join(args.space_dir, module_name + '.py')
        new_msg = str(e) + '. Verify that the module %s exists.' % expected_module_path
        logger.exception(new_msg)
        reraise(type(e), type(e)(
                new_msg), sys.exc_info()[2])
    logger.debug("Adding space definitions from %s" % fpath_space)
    for key, value in space_add.items():
        space_base[key] = value
    return space_base

def run(run_name, args):
    run_dir = os.path.join(args.log_dir, run_name)
    run_dir_already_exists = False
    if not os.path.isdir(run_dir):
        os.makedirs(run_dir)
    else:
        run_dir_already_exists = True
    global logger
    logger = lu.setup_logging(os.path.join(run_dir, 'log.txt'),
                              name=[args.logger_name, None][args.logger_name_none])
    if run_dir_already_exists:
        logger.debug("Found run directory %s", run_dir)
    else:
        logger.debug("Created run directory %s", run_dir)
    fpath_trials = os.path.join(run_dir, "trials.pkl")
    max_evals = args.nb_evals
    try:
        trials = pickle.load(open(fpath_trials, "rb"))
        logger.info("Loading saved trials from %s" % fpath_trials)
        max_evals += len(trials.trials)
        logger.debug("Rerunning from {} trials to add more.".format(
            len(trials.trials)))
    except:
        trials = Trials()
        logger.info("Starting trials from scratch.")
    
    cfg = load_config(args.fpath_cfg, logger)
    space = {
        'log_dir' : os.path.abspath(os.path.expanduser(args.log_dir)),
        'run_dir' : run_dir,
        'run_name': run_name,
        'base'  : cfg,
#        'learning_rate' : hp.loguniform('learning_rate', -7*np.log(10), -1*np.log(10))
#         'learning_rate' : hp.choice('learning_rate', [0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]),
        }
    space = add_space(space, args.space_dir)
    fpath_space = os.path.join(run_dir, "space.pkl")
    logger.debug("Save space to %s" % fpath_space)
    with open(fpath_space, "wb") as h:
        pickle.dump(space, h)
    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,
                trials=trials,
                max_evals=max_evals)
#    print(best)
    fpath_best_results = os.path.join(run_dir, 'best_results.txt.json')
    logger.info("Best results saved to %s" % fpath_best_results)
    best_eval = space_eval(space, best)
    logger.info("Best results obtained via %s" % best_eval)
    with open(fpath_best_results, "wb") as h:
        json.dump(best_eval, h)
    logger.info("Save trials to %s" % fpath_trials)
    with open(fpath_trials, "wb") as h:
        pickle.dump(trials, h)
    logger.info("Done.")

def handleArgs(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config",
                        dest="fpath_cfg", type=str, required=True,
                        help="Path to master config file")
    parser.add_argument("--log_dir", dest="log_dir", type=str,
                        default=os.path.abspath(os.path.curdir),
                        help="Set parent log directory for all runs")
    parser.add_argument("--logger_name", dest="logger_name", type=str,
                        default=__name__,
                        help="Set name for process logging")
    parser.add_argument('--logger_name_none', action='store_true')
    parser.add_argument("--run_name", dest="run_name", type=str,
                        default=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                        help="Set name for run")    
    parser.add_argument("--run_name_prefix", dest="run_name_prefix", type=str,
                        default='',
                        help="Set prefix run name")
    parser.add_argument("--nb_evals", dest="nb_evals", type=int, required=True,
                        help="Maximum number of evaluations")
    parser.add_argument("--space_dir", dest="space_dir", type=str,
                        required=True,
                        help="Path to directory with module for space definition")
    
    return parser.parse_args(args=args)

if __name__ == '__main__':
    args = handleArgs()
    run(args.run_name_prefix + args.run_name,
        args
        )
    pass
