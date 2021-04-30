import os
import pathlib
import random
from itertools import product

import fire
import numpy as np
import pandas as pd
from configs import ROOT_DIR
from experiments import AGENT_DICT
from fastprogress.fastprogress import master_bar, progress_bar

MAX_EVALS = 200
count = 0

EXP_DIR = "tmaze"
JOBS_DIR = "jobs"


cur_dir = pathlib.Path(os.path.split(os.path.realpath(__file__))[0])

def create_job(agent_type, hyper_params):
    global count
    cmd = f"python -m experiments.{EXP_DIR}.run_single_job --agent_type=\"{agent_type}\" --hyper_params=\"{hyper_params}\""
    with open(cur_dir/f"jobs/tasks_{count}.sh", 'w') as f:
        f.write(cmd)
    print(count, cmd)
    count += 1


def random_search(agent_type, param_grid, mb=master_bar(range(1)), max_evals=MAX_EVALS):
    """Random search for hyperparameter optimization"""

    # Keep searching until reach max evaluations
    for j in mb:
        for i in progress_bar(range(max_evals),parent=mb):

            # Choose random hyperparameters
            hyper_params = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}
            print(hyper_params)
            # Evaluate randomly selected hyperparameters
            create_job(agent_type, hyper_params)


def grid_search(agent_type, param_grid):
    """grid search for hyperparameter optimization"""
    param_keys, values = zip(*param_grid.items())

    param_combos = [dict(zip(param_keys, combo)) for combo in product(*values)]

    mb = master_bar(param_combos)

    for i, hyper_params in enumerate(mb):
        print(hyper_params)
        create_job(agent_type, hyper_params)

agents = {
    "GRU": None,
    "ESN_V2": None,
}


def get_lr(b=1e-2, a=2, n=5):
    return list(b/a**np.array(list(range(0, n))))


params_to_search = {
    "RNN": {
        "step_size": get_lr(n=6),
    },
    "GRU": {
        "step_size": get_lr(n=6),
    },
    "Trace": {
        "step_size": get_lr(n=6),
        "alpha": get_lr(b=0.4, n=4),
    },
    "ESN_V2": {
        "step_size": get_lr(n=6),
        "beta": [0.01, 0.05, 0.1, 0.15, 0.2, 0.25],
    },
}

def write_jobs(append=True, agents=None):
    (cur_dir/f"{JOBS_DIR}").mkdir(parents=True, exist_ok=True)
    cur_tsk_fs = [f for f in os.listdir(f"experiments/{EXP_DIR}/{JOBS_DIR}") if f.startswith("tasks")]

    if append:
        global count
        count = len(cur_tsk_fs)
    else:
        for fname in cur_tsk_fs:
            os.remove(cur_dir/f"{JOBS_DIR}"/fname)

    bgn_count = count

    if agents is None:
        agents = list(params_to_search.keys())
    elif not isinstance(agents, list):
        agents = [agents]

    for agent_type in master_bar(agents):
        if agent_type not in list(AGENT_DICT.keys()):
            print(f"{agent_type} is not found in experiments/__init__.py; skipping")
            continue
        print(agent_type)
        if agent_type == 'XXX':
            random_search(agent_type, params_to_search[agent_type], max_evals=100)
        elif agent_type in ("XXX", "XXX"):
            random_search(agent_type, params_to_search[agent_type])
        else:
            grid_search(agent_type, params_to_search[agent_type])
    print(f'Jobs: sbatch --array={bgn_count}-{count-1} ./experiments/{EXP_DIR}/{JOBS_DIR}/run_cpu.sh')

if __name__ == "__main__":
    fire.Fire(write_jobs)
