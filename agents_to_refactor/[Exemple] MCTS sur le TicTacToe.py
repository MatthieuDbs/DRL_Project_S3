import math
import os

from matplotlib import pyplot as plt

from tqdm import tqdm
import time

import numpy as np


class MCTSSingleAgentEnv:

    def state_id(self) -> int:
        pass

    def max_actions_count(self) -> int:
        pass

    def is_game_over(self) -> bool:
        pass

    def act_with_action_id(self, action_id: int):
        pass

    def score(self) -> float:
        pass

    def available_actions_ids(self) -> np.ndarray:
        pass

    def reset(self):
        pass

    def view(self):
        pass

    def clone(self) -> 'MCTSSingleAgentEnv':
        pass


class TTTVsRandom(MCTSSingleAgentEnv):
    def __init__(self):
        self.grid = np.zeros((3, 3))
        self.game_over = False
        self.aa = [i for i in range(9)]
        self.current_score = 0.0
        self.id = 0

    def state_id(self) -> int:
        return self.id

    def max_actions_count(self) -> int:
        return 9

    def is_game_over(self) -> bool:
        return self.game_over

    def act_with_action_id(self, action_id: int):
        assert (not self.is_game_over())
        row = action_id // 3
        col = action_id % 3
        assert (self.grid[row, col] == 0)

        self.grid[row, col] = 1
        self.id += 3 ** action_id * 1
        self.aa.remove(action_id)

        if self.grid[row, 0] == self.grid[row, 1] and self.grid[row, 0] == self.grid[row, 2] or \
                self.grid[0, col] == self.grid[1, col] and self.grid[0, col] == self.grid[2, col] or \
                self.grid[1, 1] == self.grid[0, 0] and self.grid[1, 1] == self.grid[2, 2] and self.grid[1, 1] == 1 or \
                self.grid[2, 0] == self.grid[1, 1] and self.grid[0, 2] == self.grid[1, 1] and self.grid[1, 1] == 1:
            self.current_score = 1.0
            self.game_over = True
            return

        if len(self.aa) == 0:
            self.current_score = 0.0
            self.game_over = True
            return

        action_id = np.random.choice(self.aa)
        assert (not self.is_game_over())
        row = action_id // 3
        col = action_id % 3
        assert (self.grid[row, col] == 0)

        self.grid[row, col] = 2
        self.id += 3 ** action_id * 2
        self.aa.remove(action_id)

        if self.grid[row, 0] == self.grid[row, 1] and self.grid[row, 0] == self.grid[row, 2] or \
                self.grid[0, col] == self.grid[1, col] and self.grid[0, col] == self.grid[2, col] or \
                self.grid[1, 1] == self.grid[0, 0] and self.grid[1, 1] == self.grid[2, 2] and self.grid[1, 1] == 2 or \
                self.grid[2, 0] == self.grid[1, 1] and self.grid[0, 2] == self.grid[1, 1] and self.grid[1, 1] == 2:
            self.current_score = -1.0
            self.game_over = True
            return

        if len(self.aa) == 0:
            self.current_score = 0.0
            self.game_over = True
            return

    def score(self) -> float:
        return self.current_score

    def available_actions_ids(self) -> np.ndarray:
        return np.array(self.aa)

    def reset(self):
        self.grid = np.zeros((3, 3))
        self.game_over = False
        self.aa = [i for i in range(9)]
        self.current_score = 0.0
        self.id = 0

    def view(self):
        print(f'Score : {self.score()}')
        print(f'Game Over : {self.is_game_over()}')
        for row in range(3):
            for col in range(3):
                c = self.grid[row, col]
                if c == 0:
                    print('_', end='')
                elif c == 1:
                    print('X', end='')
                elif c == 2:
                    print('O', end='')
                else:
                    raise "WTF"
            print()
        print()

    def clone(self) -> 'MCTSSingleAgentEnv':
        cloned_env = TTTVsRandom()
        cloned_env.grid = np.copy(self.grid)
        cloned_env.aa = self.aa.copy()
        cloned_env.current_score = self.current_score
        cloned_env.game_over = self.game_over
        cloned_env.id = self.id

        return cloned_env


def monte_carlo_tree_search_and_choose_action(env: MCTSSingleAgentEnv,
                                              iteration_count: int = 200) -> int:
    tree = {}

    root = env.state_id()
    tree[root] = {}
    for a in env.available_actions_ids():
        tree[root][a] = {
            'mean_score': 0.0,
            'selection_count': 0,
            'consideration_count': 0,
        }

    for _ in range(iteration_count):
        cloned_env = env.clone()
        current_node = cloned_env.state_id()

        nodes_and_chosen_actions = []

        # SELECTION
        while not cloned_env.is_game_over() and \
                not any(filter(lambda stats: stats['selection_count'] == 0, tree[current_node].values())):

            best_action = None
            best_action_score = None
            for (a, a_stats) in tree[current_node].items():
                ucb1_score = a_stats['mean_score'] + math.sqrt(2) * math.sqrt(
                    math.log(a_stats['consideration_count']) / a_stats['selection_count'])
                if best_action_score is None or ucb1_score > best_action_score:
                    best_action = a
                    best_action_score = ucb1_score

            nodes_and_chosen_actions.append((current_node, best_action))
            cloned_env.act_with_action_id(best_action)
            current_node = cloned_env.state_id()

            if current_node not in tree:
                tree[current_node] = {}
                for a in cloned_env.available_actions_ids():
                    tree[current_node][a] = {
                        'mean_score': 0.0,
                        'selection_count': 0,
                        'consideration_count': 0,
                    }

        # EXPAND
        if not cloned_env.is_game_over():
            random_action = np.random.choice(list(
                map(lambda action_and_stats: action_and_stats[0],
                    filter(lambda action_and_stats: action_and_stats[1]['selection_count'] == 0,
                           tree[current_node].items())
                    )
            ))

            nodes_and_chosen_actions.append((current_node, random_action))
            cloned_env.act_with_action_id(random_action)
            current_node = cloned_env.state_id()

            if current_node not in tree:
                tree[current_node] = {}
                for a in cloned_env.available_actions_ids():
                    tree[current_node][a] = {
                        'mean_score': 0.0,
                        'selection_count': 0,
                        'consideration_count': 0,
                    }

        # EVALUATE / ROLLOUT
        while not cloned_env.is_game_over():
            cloned_env.act_with_action_id(np.random.choice(cloned_env.available_actions_ids()))

        score = cloned_env.score()

        # BACKUP / BACKPROPAGATE / UPDATE STATS
        for (node, chose_action) in nodes_and_chosen_actions:
            for a in tree[node].keys():
                tree[node][a]['consideration_count'] += 1
            tree[node][chose_action]['mean_score'] = (
                    (tree[node][chose_action]['mean_score'] * tree[node][chose_action]['selection_count'] + score) /
                    (tree[node][chose_action]['selection_count'] + 1)
            )
            tree[node][chose_action]['selection_count'] += 1

    most_selected_action = None
    most_selected_action_selection_count = None

    for (a, a_stats) in tree[root].items():
        if most_selected_action_selection_count is None or a_stats[
            'selection_count'] > most_selected_action_selection_count:
            most_selected_action = a
            most_selected_action_selection_count = a_stats['selection_count']

    return most_selected_action


def run_ttt_n_games_and_return_mean_score(games_count: int) -> float:
    env = TTTVsRandom()
    total = 0.0
    wins = 0
    losses = 0
    draws = 0
    for _ in tqdm(range(games_count)):
        env.reset()

        while not env.is_game_over():
            chosen_a = monte_carlo_tree_search_and_choose_action(env)
            env.act_with_action_id(chosen_a)

        if env.score() > 0:
            wins += 1
        elif env.score() < 0:
            losses += 1
        else:
            draws += 1
        total += env.score()

    print(f"MCTS - wins : {wins}, losses : {losses}, draws : {draws}")
    print(f"MCTS - mean_score : {total / games_count}")
    return total / games_count


if __name__ == "__main__":
    run_ttt_n_games_and_return_mean_score(1000)
