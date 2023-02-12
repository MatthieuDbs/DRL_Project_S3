import math
import os

from matplotlib import pyplot as plt

from tqdm import tqdm
import time

import numpy as np

from env import DeepSingleAgentEnv, LineWorld


def monte_carlo_random_rollout_and_choose_action(env: DeepSingleAgentEnv,
                                                 simulation_count_per_action: int = 50) -> int:
    best_action = None
    best_action_average_score = None
    for a in env.available_actions_ids():
        action_score = 0.0
        for _ in range(simulation_count_per_action):
            cloned_env = env.clone()
            cloned_env.act_with_action_id(a)

            while not cloned_env.is_game_over():
                cloned_env.act_with_action_id(np.random.choice(cloned_env.available_actions_ids()))

            action_score += cloned_env.score()
        average_action_score = action_score / simulation_count_per_action

        if best_action_average_score is None or best_action_average_score < average_action_score:
            best_action = a
            best_action_average_score = average_action_score
    return best_action


def run_ttt_n_games_and_return_mean_score(games_count: int) -> float:
    env = LineWorld(10)
    total = 0.0
    for _ in tqdm(range(games_count)):
        env.reset()

        while not env.is_game_over():
            chosen_a = monte_carlo_random_rollout_and_choose_action(env)
            env.act_with_action_id(chosen_a)

        total += env.score()
    return total / games_count


if __name__ == "__main__":
    print(run_ttt_n_games_and_return_mean_score(100))