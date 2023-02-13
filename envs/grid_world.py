import math

import matplotlib.pyplot as plt
import numpy as np
import numpy.random
from tqdm import tqdm
from .env_base import DeepSingleAgentEnv


BOARD_ROWS = 3
BOARD_COLS = 4
WIN_STATE = (0, 3)
LOSE_STATE = (1, 3)
START = (2, 0)
HOLE = [(1, 1)]

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class GridWorld(DeepSingleAgentEnv):
    def __init__(self, rows: int = BOARD_ROWS, cols: int = BOARD_COLS, state = START):
        self.board = np.zeros([rows, cols])
        self.cols = cols
        self.rows = rows
        for hole in HOLE:
          self.board[hole[0], hole[1]] = -1
        self.step_count = 0
        self.state = state
        self.current_score = 0
        self.id = 0

    def state_id(self):
        return self.id

    def max_action_count(self) -> int:
        return 4

    def state_description(self) -> np.ndarray:
        return np.array([(self.state[0] * self.cols + self.state[1]) / (self.cols * self.rows - 1) * 2.0 - 1.0])

    def state_dim(self) -> int:
        return 1

    def is_game_over(self) -> bool:
        if (self.state == WIN_STATE) or (self.state == LOSE_STATE):
          self.current_score = 1 if self.state == WIN_STATE else 0
          return True
        return False

    def act_with_action_id(self, action_id: int):
        self.step_count += 1
        """
        action: up, down, left, right
        -------------
        0 | 1 | 2| 3|
        1 |
        2 |
        return next position on board
        """
        self.id += (3 ** action_id) + 1

        if action_id == UP:
            nxtState = (self.state[0] - 1, self.state[1])
        elif action_id == DOWN:
            nxtState = (self.state[0] + 1, self.state[1])
        elif action_id == LEFT:
            nxtState = (self.state[0], self.state[1] - 1)
        else:
            nxtState = (self.state[0], self.state[1] + 1)

        if (nxtState[0] >= 0) and (nxtState[0] < self.rows):
            if (nxtState[1] >= 0) and (nxtState[1] < self.cols):
                if nxtState not in HOLE:
                  self.board[nxtState[0]][nxtState[1]] = 9
                  self.board[self.state[0]][self.state[1]] = 0
                  self.state = nxtState

    def score(self) -> float:
        return self.current_score

    def available_actions_ids(self) -> np.ndarray:
        return [UP, RIGHT, DOWN, LEFT]

    def reset(self):
        self.state = START
        self.current_score = 0
        self.board = np.zeros([self.rows, self.cols])
        for hole in HOLE:
          self.board[hole[0], hole[1]] = -1
        self.step_count = 0
        self.id = 0

    def view(self):
        print(f'Game Over: {self.is_game_over()}')
        print(f'score : {self.score()}')
        for row in self.board:
            for col in row:
              case = 'X' if col == 9 else 'O' if col == -1 else '_'
              print(f'{case}', end="")
            print()

    def copy(self):
        copy = GridWorld()
        copy.board = self.board.copy()
        copy.cols = self.cols
        copy.rows = self.rows
        copy.step_count = self.step_count
        copy.state = self.state
        copy.current_score = self.current_score
        copy.id = self.id

        # print("copy")
        # copy.view()

        return copy
    
    def clone(self):
        return self.copy()
