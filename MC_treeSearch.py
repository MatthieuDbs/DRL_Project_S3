import math
import time

import numpy as np
from tqdm import tqdm  

from envs import LineWorld, TTTVsRandom, Pacman, GridWorld
import numpy as np

def monte_carlo_tree_search(env,
                            iterations_count: int, deep):
    tree = {}
    root = env.state_id()
    c = math.sqrt(2)

    tree[root] = {}
    for a in env.available_actions_ids():
        tree[root][a] = [0.0, 0, 0]


    for it in range(iterations_count):
        cloned_env = env.clone()
        current_node = root

        chosen_nodes_and_actions = []

        tdeep = deep
        # SELECT
        while current_node in tree and not cloned_env.is_game_over() and not any(filter(lambda stats: stats[2] == 0, tree[current_node].values())) and tdeep > 0:
            ucb_scores = [(a, stats[0] + c*math.sqrt(math.log(stats[1]) / stats[2]))  for (a, stats) in tree[current_node].items()]

            best_a = None
            best_score = None
            for (a, ucb_score) in ucb_scores:
                if best_a is None or ucb_score > best_score:
                    best_a = a
                    best_score = ucb_score

            chosen_nodes_and_actions.append((current_node, best_a))
            cloned_env.act_with_action_id(best_a)
            current_node = cloned_env.state_id()

            if current_node not in tree:
                tree[current_node] = {}
                for a in cloned_env.available_actions_ids():
                    tree[current_node][a] = [0.0, 0, 0]
            tdeep -= 1

        # EXPAND
        if not cloned_env.is_game_over():
            a = np.random.choice(list(map(lambda kv: kv[0], filter(lambda kv: kv[1][2] == 0, tree[current_node].items()))))

            chosen_nodes_and_actions.append((current_node, a))
            cloned_env.act_with_action_id(a)
            current_node = cloned_env.state_id()

            if current_node not in tree:
                tree[current_node] = {}
                for a in cloned_env.available_actions_ids():
                    tree[current_node][a] = [0.0, 0, 0]

        # EVALUATE
        tdeep = deep
        while not cloned_env.is_game_over() and tdeep > 0:
            cloned_env.act_with_action_id(np.random.choice(cloned_env.available_actions_ids()))
            tdeep -= 1

        score = cloned_env.score()

        # BACKPROPAGATE
        for (node, a) in chosen_nodes_and_actions:
            for k in tree[node].keys():
                tree[node][k][1] += 1
            tree[node][a][0] = (tree[node][a][0] * tree[node][a][2] + score) / (tree[node][a][2] + 1)
            tree[node][a][2] += 1

    best_a = None
    best_selection_count = None
    for (a, selections) in tree[root].items():
        if best_selection_count is None or best_selection_count < selections:
            best_a = a
            best_selection_count = selections

    return best_a


def run_n_games_and_report_score(env, num_games: int):
    total_score = 0
    step_avg = 0
    time_avg = 0
    deep = 30000
    for _ in tqdm(range(num_games)):
        env.reset()
        step = 0
        _time = time.time()
        while not env.is_game_over() and step <= deep:
            a = monte_carlo_tree_search(env, 100, deep)
            env.act_with_action_id(a)
            step += 1
        _time = time.time() - _time
        time_avg += _time
        step_avg += step
        total_score += env.score()
    print (f"score: {total_score / num_games}\ntime: {time_avg / num_games}\nstep: {step_avg / num_games}")
    return total_score / num_games


env = GridWorld()
print(run_n_games_and_report_score(env, 1000))
