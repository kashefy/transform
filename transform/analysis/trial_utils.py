'''
Created on Mar 15, 2018

@author: kashefy
'''
import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from hyperopt import hp, space_eval, Trials

def load_trials(p):
    with open(p, 'rb') as h:
        trials = pickle.load(h)
        return trials
    
def load_space(p):
    with open(p, 'rb') as h:
        sp = pickle.load(h)
        return sp
    
def get_space_vals_sorted(key, trials):
    vals = set()
    for t in trials:
        for x in t.trials:
            vals.add(x['result']['space'][d])
    sorted(vals)
    return vals

def trials2DataFrame(fpath_list,
                     space_dim_names = [
                         'learning_rate',
                         'lambda_l2', 'lambda_l1',
                         'n_nodes', 'lambda_c_orientation', 'lambda_c_recognition',
                         'branch',
                      ]
                    ):
    frames = []
    for p in fpath_list:
        df = pd.DataFrame({})
        t = load_trials(p)
        run_name = os.path.basename(os.path.dirname(p))
        print('%s: %d trials.' %(p, len(t)))
#         print(t.trials[0].keys())
#         print(t.trials[0]['result']['space'].keys())
    #     print([x['result']['performance'] for x in t.trials])
    #     print([x['book_time'] for x in t.trials[:10]])
    #     print([x['book_time'] for x in t.trials[:-10]])
        for perf_name in ['performance', 'performance_orient']:
            perf = [x['result'][perf_name] for x in t.trials if perf_name in x['result']]
            if len(perf) == 0:
                perf = [np.nan]*len(t.trials)
            df[perf_name] = perf
        df['run_name'] = [run_name]*len(t.trials)
        df['task_recognition'] = [x['result']['space']['do_task_recognition'] if 'do_task_recognition' in x['result']['space']
         else 'recognition' in x['result']['space']['tasks'] if 'tasks' in x['result']['space']
         else not run_name.endswith('_o') or run_name.endswith('_rm')
                                  for x in t.trials]
        df['task_orientation'] = [x['result']['space']['do_task_orientation'] if 'do_task_orientation' in x['result']['space']
         else 'orientation' in x['result']['space']['tasks'] if 'tasks' in x['result']['space']
         else run_name.endswith('_o') or run_name.endswith('_ro')
                                  for x in t.trials]
        is_augmented = run_name.endswith('ar') or 'rm_' in run_name
        df['is_augmented'] = [is_augmented]*len(t.trials)
        is_pretrained = run_name.endswith('ar') or 'a' in run_name
        df['is_pretrained'] = [is_pretrained]*len(t.trials)
        for dname in space_dim_names:
            if 'lambda_c_' in dname:
                x = [x['result']['space'].get(dname, 1.) for x in t.trials]
            else:
                x = [x['result']['space'][dname] for x in t.trials]
            x = [str(a) if isinstance(a, (list, tuple)) and not isinstance(a, basestring) and len(a) > 0 else a for a in x]
            if dname == 'n_nodes' and (len(x) == 0 or (len(x) > 0 and x == [[]]*len(x))):
                x = [x['result']['space']['stack_dims'] for x in t.trials]
                x = [str(a) if isinstance(a, (list, tuple)) and not isinstance(a, basestring) else a for a in x]
            df[dname] = x
            assert(len(x))
        # handle case of categorical n_nodes
        df['n_nodes'] = df['n_nodes'].apply(lambda x: str([int(a) for a in str(x).replace(' ', '').replace('(', '').replace(')', '').replace('[', '').replace(']', '').split(',')]))
        df['depth'] = df['n_nodes'].apply(lambda x: len([int(a) for a in re.sub('[\[\]]', '', x).split(',')]))
        if 'num_weights' not in df:
            def count_weights(a):
    #             print(a)
                a = [int(x.strip()) for x in a.replace('(', '').replace(')', '').replace('[', '').replace(']', ',').split(',') if len(x) > 0]
                return sum([el_x * el_y for el_x, el_y in zip(a[:-1], a[1:])])
        #     df['num_weights'] = [count_weights(a) for a in df['n_nodes']]
    #         df['num_weights'] = df['n_nodes'].apply(lambda x: count_weights(x))
            df['num_weights'] = df['n_nodes'].apply(lambda x: count_weights(x))
        frames.append(df)
    df = pd.concat(frames)
    return df
