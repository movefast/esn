import os
import pathlib
import time

import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
from agents.columnar.columnar_agent_stable_v1_base import \
    RNNAgent as ESNAgentV2
from agents.gru_agent import RNNAgent as GRUAgent
# from agents.rnn_agent import RNNAgent as RNNAgent
from configs import ROOT_DIR
from env.gridworld import MazeEnvironment
from env.tmaze import MazeEnvironment as TMazeEnvironment
from fastprogress.fastprogress import master_bar, progress_bar
from tqdm import tqdm

exp_decay_explor = True
num_runs = 25
tot_num_steps = 10000
# -----> gamma
gamma = 0.9

cur_dir = pathlib.Path(os.path.split(os.path.realpath(__file__))[0])


def run_episode(env, agent, state_visits=None, keep_history=False):
    is_terminal = False
    sum_of_rewards = 0
    step_count = 0

    obs = env.env_start(keep_history=keep_history)
    action = agent.agent_start(obs)

    if state_visits is not None:
        state_visits[obs[0]] += 1

    while not is_terminal:
        reward, obs, is_terminal = env.env_step(action)
        print(agent.steps,end='\r')
        sum_of_rewards += reward
        step_count += 1
        state = obs
        if step_count == 500:
            agent.agent_end(reward, state, append_buffer=False)
            break
        elif is_terminal:
            agent.agent_end(reward, state, append_buffer=True)
        else:
            action = agent.agent_step(reward, state)

        if state_visits is not None:
            state_visits[state[0]] += 1

    if keep_history:
        history = env.history
        env.env_cleanup()
        return sum_of_rewards, step_count, history
    else:
        return sum_of_rewards, step_count


agents = {
    "GRU": GRUAgent,
    "ESN_V2": ESNAgentV2,
}


agent_infos = {
    "GRU": {"step_size": 0.0025},
    "SubsampleAgent": {"step_size": 0.005},
    "ESN_V2": {"step_size": 0.00125, "beta":0.15},
}

envs = {
    # 'Grid-World': MazeEnvironment,
    'TMaze': TMazeEnvironment,
}

maze_config = [
    "XOOOXXXXXO",
    "OOXOOOOOOO",
    "XOOOXXXXXO"
]

env_infos = {
    'tmaze1': {
        "maze_dim": [len(maze_config), len(maze_config[0])],
        "start_state": [1, 0],
        "end_states": [[0, len(maze_config[0])-1], [2,len(maze_config[0])-1]],
        "obstacles":np.asarray(np.where(np.array([list(row) for row in maze_config]) == 'X')).T.tolist(),
    }
}

all_state_visits = {}  # Contains state visit counts during the last 10 episodes
all_history = {}
metrics = {"cum_rewards_per_run": {}, "hyper_params": {}, "sampled_states_per_run": {}}


def objective(agent_type, hyper_params, num_runs=num_runs):
    start = time.time()

    Environment = envs['TMaze']

    all_history = {}

    mb = master_bar(env_infos.items())
    algorithm = agent_type + '_' + '_'.join([f'{k}_{v}' for k, v in hyper_params.items()])
    print(algorithm)

    for env_name, env_info in mb:
        print(env_name)

        for metric in metrics:
            if env_name not in metrics[metric]:
                metrics[metric][env_name] = {}
                metrics[metric][env_name].setdefault(algorithm, [])

        for run in progress_bar(range(num_runs), parent=mb):
            agent = agents[agent_type]()
            env = Environment()
            env.env_init(env_info)
            # print(env_info)
            # -----> gamma seed
            agent_info = {"num_actions": 4, "num_states": env.cols * env.rows, "epsilon": 1, "step_size": 0.1, "discount": gamma}
            agent_info["seed"] = run
            agent_info.update(agent_infos[agent_type])
            agent_info.update(hyper_params)
            np.random.seed(run)
            agent.agent_init(agent_info)

            cum_rewards = [0]
            sampled_states_per_ep = []
            if exp_decay_explor:
                epsilon = 1
            else:
                epsilon = .1

            cur_step = 0
            cur_episode = 0
            while cur_step < tot_num_steps:
                print(f"episode {cur_episode}",end='\r')
                agent.epsilon = epsilon
                episode_start = time.time()
                ep_return, ep_num_steps = run_episode(env, agent, keep_history=False)
                episode_end = time.time()
                # print(f"episode {cur_episode} took {episode_end-episode_start} seconds to complete",end='\r')
                # all_history.setdefault(env_name, {}).setdefault(algorithm, []).append(history)
                # -----> epsilon
                if exp_decay_explor:
                    epsilon *= 0.99

                # -----> todo add sampled state to agents
                # sampled_states_per_ep.append(agent.sampled_state.tolist())
                # agent.sampled_state = np.zeros(env.cols * env.rows)

                cur_episode += 1
                # -----> maybe don't need while clause anymore
                cur_step += ep_num_steps
                # -----> could refactor to let run_episode return a list of rewards and saves the trouble here
                if cur_step > tot_num_steps:
                    cum_rewards.extend([cum_rewards[-1]]*(ep_num_steps - (cur_step - tot_num_steps)))
                else:
                    cum_rewards.extend([cum_rewards[-1]]*(ep_num_steps-1) + [ep_return])

            metrics["cum_rewards_per_run"][env_name].setdefault(algorithm, []).append(cum_rewards)
            metrics["sampled_states_per_run"][env_name].setdefault(algorithm, []).append(sampled_states_per_ep)

    end = time.time()
    print("total run time: ", end - start)
    metrics['hyper_params'][algorithm] = hyper_params
    (cur_dir/"metrics").mkdir(parents=True, exist_ok=True)
    torch.save(metrics, cur_dir/f'metrics/{algorithm}.torch')
    return algorithm, np.mean(metrics["cum_rewards_per_run"][env_name][algorithm]), hyper_params

if __name__ == '__main__':
    fire.Fire(objective)
