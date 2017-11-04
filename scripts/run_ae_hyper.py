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
import tempfile
import pprint
import yaml
import numpy as np
from hyperopt import hp, fmin, tpe, space_eval, Trials
from hyperopt import STATUS_OK, STATUS_FAIL
from hyperopt import pyll
import transform.logging_utils as lu
from transform.cfg_utils import load_config
import run_autoencoder as script
logger = None

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

def get_items_suffix(items):
    return '_'.join(['-'.join([k, str(v)]) for k, v in items])

def unique_dirname(dir_):
    dir_parent = os.path.dirname(dir_)
    prefix = os.path.basename(dir_) + '_'
    dir_ = tempfile.mkdtemp(suffix='', prefix=prefix, dir=dir_parent)
    logger.debug("Created directory %s", dir_)
    return dir_

def update_cfg(base, params):
    excl = ['base', 'run_dir', 'run_name', 'log_dir', 'pass_on_args']
    items_new = []
    for key, value in params.items():
        if key not in excl:
            base[key] = value
            items_new.append([key, value])
    base['data_seed'] = np.random.randint(3)
    items_new.append(['data_seed', base['data_seed']])
    base['do_augment_rot'] = [False, True][np.random.randint(2)]
    items_new.append(['do_augment_rot', base['do_augment_rot']])
    for key, value in base.items():
        if isinstance(value, tuple):
            base[key] = list(value)
    suffix = params['run_name'] + '_' + get_items_suffix(items_new).replace(' ', '')
    run_dir_new = unique_dirname(os.path.join(params['run_dir'], suffix))
    base['log_dir'] = run_dir_new # keep log and run dirs the same
    items_new.append(['log_dir', base['log_dir']])
    base['run_dir'] = run_dir_new
    items_new.append(['run_dir', base['run_dir']])
    return base, items_new, suffix

#def objective(params_all):
def objective(params):
    status = STATUS_OK
    fpath_cfg_dst_list = []
#    for space_idx in xrange(params_all['num_spaces']):
#        params = {k[:k.index('spaceIdx-%d' % space_idx):v] for k,v in params_all.items()}
    for cfg_idx, base in enumerate(params['base']):
        cfg, _, suffix = update_cfg(base, params)
        fpath_cfg_dst = os.path.join(params['run_dir'], 'config_%s_%d-%s.yml' % (suffix, cfg_idx, cfg['prefix']))
        logger.debug("Write config %d/%d %s" % (cfg_idx, len(params['base']), fpath_cfg_dst))
        with open(fpath_cfg_dst, 'w') as h:
            h.write(yaml.dump(cfg))
            fpath_cfg_dst_list.append(fpath_cfg_dst)
    ch_args_in = [it for sublist in [['-c', p] for p in fpath_cfg_dst_list] for it in sublist]
    ch_args_in.extend([
                      '--log_dir', cfg['log_dir'],
                      '--run_name', suffix,
                      '--run_dir', cfg['run_dir'],
                      ])
#    ch_args_in = ['-c', fpath_cfg_dst,
#                  '-c', fpath_cfg_dst,
#                  '--log_dir', cfg['log_dir'],
#                  '--run_name', suffix,
#                  '--run_dir', cfg['run_dir'],
#                  ]
    for k, v in params['pass_on_args']:
        logger.debug("Passing on %s %s" % (k, str(v)))
        ch_args_in.extend([k, str(v)])
    args_ch = script.handleArgs(args=ch_args_in)
    result_ae, result_mlp, result_mlp_fine = \
        script.run(args_ch.run_name,
           args_ch
           )
    if result_ae is None:
        status = STATUS_FAIL
        logger.error("fmin failed with params: %s" % params)
    result = {
        "name"      : result_mlp_fine.name,
        "loss"      : -result_mlp_fine.max, 
        "performance" : result_mlp_fine.max, 
        "status"    : status,
        "space"     : cfg,
    }
    return result
    
def add_space(space_base, space_dir, do_reload=False):
    if space_dir not in sys.path:
        sys.path.insert(0, space_dir) # prepend to path
    try:
        import space
        if do_reload:
            reload(space)
        fpath_space = space.__file__
        if os.path.splitext(fpath_space)[-1] == '.pyc':
            os.remove(fpath_space)
            reload(space)
        space_add = space.space
    except ImportError as e:
        module_name = str(e).split(' ')[-1]
        expected_module_path = os.path.join(space_dir, module_name + '.py')
        new_msg = str(e) + '. Verify that the module %s exists.' % expected_module_path
        logger.exception(new_msg)
        reraise(type(e), type(e)(
                new_msg), sys.exc_info()[2])
    logger.debug("Adding space definitions from %s" % fpath_space)
    for key, value in space_add.items():
        for base_idx in range(len(space_base['base'])):
            space_base['base'][base_idx].pop(key, None)
            space_base[key] = value
    return space_base

def init_trials(fpath_trials, force_fresh=False):
    if force_fresh:
        try:
            trials = pickle.load(open(fpath_trials, "rb"))
            logger.info("Loading saved trials from %s" % fpath_trials)
            logger.debug("Rerunning from {} trials to add more.".format(
                len(trials.trials)))
        except:
            trials = Trials()
            logger.info("Starting trials from scratch.")
    else:
        trials = Trials()
        logger.info("Starting trials from scratch.")
    return trials
        
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
    logger = lu.setup_logging(os.path.join(run_dir, 'log.txt'),
                              name=[args.logger_name, None][args.logger_name_none])
    if run_dir_already_exists:
        logger.debug("Found run directory %s", run_dir)
    else:
        logger.debug("Created run directory %s", run_dir)
    fpath_trials = os.path.join(run_dir, "trials.pkl")
    trials = init_trials(fpath_trials, force_fresh=args.force_fresh_trials)
    max_evals = args.nb_evals + len(trials.trials)

#    space_all = {}
#    for cfgidx, (fpath_cfg, space_dir) in enumerate(zip(args.fpath_cfg_list, args.space_dir)):
    space_dir = args.space_dir
    space = {
        'log_dir' : os.path.abspath(os.path.expanduser(args.log_dir)),
        'run_dir' : run_dir,
        'run_name': run_name,
        'base'  : [load_config(fpath_cfg, logger) for fpath_cfg in args.fpath_cfg_list],
        'pass_on_args' : [
            ['--per_process_gpu_memory_fraction', args.per_process_gpu_memory_fraction],
            ['--data_dir', args.data_dir],
            ],
        'tf_record_prefix'  : args.tf_record_prefix,
    }
#    space = add_space(space, space_dir, do_reload=cfgidx>0)
#    fpath_space = os.path.join(run_dir, "space_%d.pkl" % (cfgidx,))
    space = add_space(space, space_dir)
    fpath_space = os.path.join(run_dir, "space.pkl")
    logger.debug("Save space to %s" % fpath_space)
    with open(fpath_space, "wb") as h:
        pickle.dump(space, h)
#    for k, v in space.items():
#        space_all[k+'_spaceIdx-%d' % cfgidx] = v
    pp = pprint.PrettyPrinter(indent=4, width=100)
#    for _ in range(10):
#        pp.pprint(pyll.stochastic.sample(space))
#    space_all['num_spaces'] = len(args.fpath_cfg_list)
    pp.pprint(space)
    best = fmin(fn=objective,
#                space=space_all,
                space=space,
                algo=tpe.suggest,
                trials=trials,
                max_evals=max_evals)
    fpath_best_results = os.path.join(run_dir, 'best_results.txt.json')
    logger.info("Best results saved to %s" % fpath_best_results)
    best_eval = space_eval(space, best)
    logger.info("Best results obtained via %s" % best_eval)
    with open(fpath_best_results, "wb") as h:
        json.dump(best_eval, h)
    logger.info("Save %d trials to %s" % (len(trials), fpath_trials))
    with open(fpath_trials, "wb") as h:
        pickle.dump(trials, h)
    logger.info("Finished hyper run %s" % run_name)

def handleArgs(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", action='append',
                        dest="fpath_cfg_list", type=str, required=True,
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
    parser.add_argument("--space_dir", action='store',
                        dest="space_dir", type=str, required=True,
                        help="Path to directory with module for space definition")
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
    parser.add_argument("--force_fresh_trials",  action='store_true',
                        dest="force_fresh_trials",
                        help="Force to start new set of trials and not resume existing ones.")
    return parser.parse_args(args=args)

if __name__ == '__main__':
    args = handleArgs()
    run(args.run_name_prefix + args.run_name,
        args
        )
    pass
