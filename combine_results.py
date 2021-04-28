import collections
import glob
from datetime import date

import fire
import numpy as np
import torch

from configs import ROOT_DIR

SKIP_METRICS = ["sampled_states_per_run"]


def dict_merge(dct, merge_dct):
    """ Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
    updating only top-level keys, dict_merge recurses down into dicts nested
    to an arbitrary depth, updating keys. The ``merge_dct`` is merged into
    ``dct``.
    :param dct: dict onto which the merge is executed
    :param merge_dct: dct merged into dct
    :return: None
    """
    for k, v in merge_dct.items():
        if (k in dct and isinstance(dct[k], dict)
                and isinstance(merge_dct[k], collections.Mapping)):
            if k in SKIP_METRICS:
                continue
            dict_merge(dct[k], merge_dct[k])
        elif k in dct and isinstance(dct[k], list) and isinstance(v, list):
            print('hehehe')
        else:
            dct[k] = merge_dct[k]

def combine_results(exp, file_name):
    today = date.today().strftime("%m_%d")
    metrics = {"cum_rewards_per_run": {}, "hyper_params": {}, "sampled_states_per_run": {}}
    for file in glob.glob(str(ROOT_DIR/f'experiments/{exp}/metrics/*')):
        dict_merge(metrics, torch.load(file))
    # 1) torch.save
    # torch.save(metrics, ROOT_DIR/f'metrics_{today}_{file_name}.torch')
    # 2) joblib.dump for large files
    import joblib
    joblib.dump(metrics, ROOT_DIR/f'metrics_{today}_{file_name}.torch')

if __name__ == '__main__':
    fire.Fire(combine_results)
