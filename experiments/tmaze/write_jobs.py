import pathlib
import random
from itertools import product

import numpy as np
import pandas as pd
from configs import ROOT_DIR
from fastprogress.fastprogress import master_bar, progress_bar

MAX_EVALS = 200
# -----> change to take count as input in order to write jobs incrementally
count = 0


cur_dir = ROOT_DIR/"experiments/tmaze"

def create_job(agent_type, hyper_params):
    global count
    cmd = f"python -m maze.run_single_job --agent_type=\"{agent_type}\" --hyper_params=\"{hyper_params}\""
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
    "GRU": {
        "step_size": get_lr(n=6),
    },
    "ESN_V2": {
        "step_size": get_lr(n=6),
        "beta": [0.01, 0.05, 0.1, 0.15, 0.2, 0.25],
    },
}


if __name__ == "__main__":
    (cur_dir/"jobs").mkdir(parents=True, exist_ok=True)
    for agent_type in master_bar(list(agents.keys())):
        print(agent_type)
        if agent_type == 'XXX':
            random_search(agent_type, params_to_search[agent_type], max_evals=100)
        elif agent_type in ("XXX", "XXX"):
            random_search(agent_type, params_to_search[agent_type])
        else:
            grid_search(agent_type, params_to_search[agent_type])

        print('Jobs: sbatch --array={}-{} ./experiments/tmaze/jobs/run_cpu.sh'.format(0, count-1))

