import numpy as np
import tensorflow as tf
import random
# from envs import DeepSingleAgentEnv
from .env_base import DeepSingleAgentEnv

class TTTVsRandom(DeepSingleAgentEnv):
    def __init__(self):
        self.grid = np.zeros((3, 3))
        self.game_over = False
        self.aa = [i for i in range(9)]
        self.current_score = 0.0

    def state_description(self) -> np.ndarray:
        return tf.keras.utils.to_categorical(self.grid, 3).flatten()

    def state_dim(self) -> int:
        return 27

    def max_action_count(self) -> int:
        return 9

    def is_game_over(self) -> bool:
        return self.game_over

    def act_with_action_id(self, action_id: int):
        assert (not self.is_game_over())
        row = action_id // 3
        col = action_id % 3
        assert (self.grid[row, col] == 0)

        self.grid[row, col] = 1
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

    def copy(self):
        copy = TTTVsRandom()
        copy.grid = self.grid.copy()
        copy.game_over = self.game_over
        copy.aa = self.aa.copy()
        copy.current_score = self.current_score

        return copy

    def get_grid(self):
        return self.grid

