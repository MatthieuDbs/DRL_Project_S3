from enum import Enum
import numpy as np
import tensorflow as tf
import random
from .env_base import DeepSingleAgentEnv

#supergum efect count (1 by turn)
COUNT = 30

#pacman codded by turn because is more easy for manipulate with agents
#TODO: replace with realtime pacman

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

BASE_GRID = [
          [1 for _ in range(28)],
          [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1],
          [1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1],
          [1, 3, 1, 0, 0, 1, 2, 1, 0, 0, 0, 1, 2, 1, 1, 2, 1, 0, 0, 0, 1, 2, 1, 0, 0, 1, 3, 1],
          [1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1],
          [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1],
          [1, 2, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 2, 1],
          [1, 2, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 2, 1],
          [1, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 1],
          [1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1],
          [0, 0, 0, 0, 0, 1, 2, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 2, 1, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 1, 2, 1, 1, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 1, 1, 2, 1, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 1, 2, 1, 1, 0, 1, 1, 1, 9, 9, 1, 1, 1, 0, 1, 1, 2, 1, 0, 0, 0, 0, 0],
          [1, 1, 1, 1, 1, 1, 2, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 2, 1, 1, 1, 1, 1, 1],
          [1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 1, 5, 0, 6, 0, 7, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 1],
          [1, 1, 1, 1, 1, 1, 2, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 2, 1, 1, 1, 1, 1, 1],
          [0, 0, 0, 0, 0, 1, 2, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 2, 1, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 1, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 1, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 1, 2, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 2, 1, 0, 0, 0, 0, 0],
          [1, 1, 1, 1, 1, 1, 2, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 2, 1, 1, 1, 1, 1, 1],
          [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1],
          [1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1],
          [1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1],
          [1, 3, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 0, 8, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 3, 1],
          [1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 1],
          [1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 1],
          [1, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 1],
          [1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1],
          [1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1],
          [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1],
          [1 for _ in range(28)]
        ]

EMP = 0
WAL = 1
GUM = 2
SGU = 3
RED = 4
BLU = 5
PIN = 6
ORA = 7
PAC = 8
GAT = 9

class Action(Enum):
    NONE = 0
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4

class Color(Enum):
  PURPLE = '\033[95m'
  BLUE = '\033[94m'
  CYAN = '\033[96m'
  GREEN = '\033[92m'
  ORANGE = '\033[33m'
  YELLOW = '\033[93m'
  RED = '\033[91m'
  CLEAR = '\033[0m'
  BOLD = '\033[1m'
  UNDERLINE = '\033[4m'

INIT_RED = {"x": 11, "y": 13}
INIT_BLUE = {'x':14, 'y':11}
INIT_PINK = {'x': 14, 'y': 13}
INIT_ORANGE = {'x': 14, 'y': 15}

class Position():
    def __init__(self, pac=(23, 14), copy  = None):
        if copy is not None:
            self.pac = Struct(x = copy.pac.x, y = copy.pac.y)
            self.ghost_r = Struct(**copy.ghost_r)
            self.ghost_b = Struct(**copy.ghost_b)
            self.ghost_p = Struct(**copy.ghost_p)
            self.ghost_o = Struct(**copy.ghost_o)
        else:
            self.pac = Struct(x = pac[0], y = pac[1])
            self.ghost_r = Struct(**INIT_RED)
            self.ghost_b = Struct(**INIT_BLUE)
            self.ghost_p = Struct(**INIT_PINK)
            self.ghost_o = Struct(**INIT_ORANGE)

    def move(self, move, entity = PAC):
        if entity == PAC:
            self.pac.x += move[0]
            self.pac.y += move[1]
        if entity == RED:
            self.ghost_r.x += move[0]
            self.ghost_r.y += move[1]
        if entity == BLU:
            self.ghost_b.x += move[0]
            self.ghost_b.y += move[1]
        if entity == PIN:
            self.ghost_p.x += move[0]
            self.ghost_p.y += move[1]
        if entity == ORA:
            self.ghost_o.x += move[0]
            self.ghost_o.y += move[1]

    def init_pos(self, entity, pos = None):
        if entity == RED:
            self.ghost_r = Struct(**INIT_RED) if pos is None else Struct(**pos)
        if entity == BLU:
            self.ghost_b = Struct(**INIT_BLUE) if pos is None else Struct(**pos)
        if entity == PIN:
            self.ghost_p = Struct(**INIT_PINK) if pos is None else Struct(**pos)
        if entity == ORA:
            self.ghost_o = Struct(**INIT_ORANGE) if pos is None else Struct(**pos)
    
    def get_pos(self, entity):
        if entity == RED:
            return self.ghost_r
        if entity == BLU:
            return self.ghost_b
        if entity == PIN:
            return self.ghost_p
        if entity == ORA:
            return self.ghost_o

terrain_dict = {
  EMP: {"c": "", "s": " "},
  WAL: {"c": Color.BLUE.value, "s": "#"},
  GUM: {"c": Color.ORANGE.value, "s" :"."},
  SGU: {"c": Color.ORANGE.value, "s" :"o"},
  RED: {"c": Color.RED.value, "s" :"G"},
  BLU: {"c": Color.CYAN.value, "s" :"G"},
  PIN: {"c": Color.PURPLE.value, "s" :"G"},
  ORA: {"c": Color.ORANGE.value, "s" :"G"},
  PAC: {"c": Color.YELLOW.value, "s" :"P"},
  GAT: {"c": "", "s": "_"},
}

act_dict = {
    Action.UP: (-1, 0),
    Action.RIGHT: (0, 1),
    Action.DOWN: (1, 0),
    Action.LEFT: (0, -1)
}

class Pacman(DeepSingleAgentEnv):
    def __init__(self):
        self.grid = BASE_GRID
        self.pos = Position()
        self.speed = (0, 0)
        self.game_over = False
        self.current_score = 0.0
        self.super = False
        self.ghost_pt = 20
        self.count = COUNT
        self.g_island = [BLU, ORA, PIN]
        self.g_ingame = [RED]
        self.g_speed = {
            RED: (0, 0),
            BLU: (0, 0),
            PIN: (0, 0),
            ORA: (0, 0)
        }
        self.g_hist = {
            RED: EMP,
            BLU: EMP,
            PIN: EMP,
            ORA: EMP
        }

    def state_dim(self) -> int:
        return 8680

    def state_description(self):
        return tf.Keras.utils.to_categorical(self.grid, 10).flatten()

    def max_action_count(self) -> int:
        return 5

    def is_game_over(self) -> bool:
        return self.game_over

    def is_gum(self) -> bool:
        hist = list(self.g_hist.values())
        if GUM in hist or SGU in hist:
            return True
        for line in self.grid:
            if GUM in line or SGU in line:
                return True
        return False

    def act_with_action_id(self, action_id: int):
        assert (not self.is_game_over())
        action = Action(action_id)
        move = act_dict[action] if action_id != 0 else self.speed
        self.speed = move

        if self.super:
            self.count -= 1
        if self.count <= 0:
            self.super = False
            self.ghost_pt = 20
            self.count = COUNT

        if self.grid[self.pos.pac.x + move[0]][self.pos.pac.y + move[1]] in [WAL, GAT]:
            move = (0,0)

        old_pos = (self.pos.pac.x, self.pos.pac.y)
        self.pos.move(move)
        new_case = self.grid[self.pos.pac.x][self.pos.pac.y]
        if new_case == GUM:
            self.current_score += 1
        if new_case in [RED, BLU, PIN, ORA]:
            if not self.super:
                self.current_score -= 180 # additional point will be count as a win game
                self.game_over = True
                return
            self.current_score += self.ghost_pt
            self.ghost_pt = self.ghost_pt * 2
            self.pos.init_pos(new_case)
            init = self.pos.get_pos(new_case)
            self.grid[init.x][init.y] = new_case
            if new_case != RED:
                self.g_ingame.remove(new_case)
                self.g_island.append(new_case)
        if new_case == SGU:
            self.current_score += 5
            self.super = True

        self.grid[old_pos[0]][old_pos[1]] = EMP
        self.grid[self.pos.pac.x][self.pos.pac.y] = PAC

        if not self.is_gum():
            self.game_over = True
            return

        #RED
        red_pos = self.pos.ghost_r
        self.move_ghost(red_pos, RED)

        #BLU
    # if BLU in self.g_ingame:
        blu_pos = self.pos.ghost_b
        self.move_ghost(blu_pos, BLU)

        #PIN
        pin_pos = self.pos.ghost_p
        self.move_ghost(pin_pos, PIN)

        #ORA
        ora_pos = self.pos.ghost_o
        self.move_ghost(ora_pos, ORA)


                
        if PIN in self.g_island and self.current_score >= 20:
            self.g_ingame.append(PIN)
            self.g_island.remove(PIN)
            old_pos = (self.pos.ghost_p.x, self.pos.ghost_p.y)
            self.pos.init_pos(PIN, {"x": 12, "y": 13})
            self.grid[old_pos[0]][old_pos[1]] = EMP
            self.grid[self.pos.ghost_p.x][self.pos.ghost_p.y] = PIN
        if BLU in self.g_island and self.current_score >= 50:
            self.g_ingame.append(BLU)
            self.g_island.remove(BLU)
            old_pos = (self.pos.ghost_b.x, self.pos.ghost_b.y)
            self.pos.init_pos(BLU, {"x": 12, "y": 13})
            self.grid[old_pos[0]][old_pos[1]] = EMP
            self.grid[self.pos.ghost_b.x][self.pos.ghost_b.y] = BLU
        if ORA in self.g_island and self.current_score >= 80:
            self.g_ingame.append(ORA)
            self.g_island.remove(ORA)
            old_pos = (self.pos.ghost_o.x, self.pos.ghost_o.y)
            self.pos.init_pos(ORA, {"x": 12, "y": 13})
            self.grid[old_pos[0]][old_pos[1]] = EMP
            self.grid[self.pos.ghost_o.x][self.pos.ghost_o.y] = ORA

    def move_ghost(self, pos, g):
        acts = self.ghost_available_actions(pos, self.g_speed[g])
        # tuple(map(lambda i, j: i + j, my_tuple_1, my_tuple_2))
        print(f"{g} pos = ({pos.x}, {pos.y}) and acts = {acts}")
        act = np.random.choice(acts)
        move = act_dict[Action(act)] if act != 0 else self.g_speed[g]

        if self.grid[pos.x + move[0]][pos.y + move[1]] in [WAL, GAT, RED, BLU, PIN, ORA]:
            move = (0,0)

        print(f"move {move}")

        old_pos = (pos.x, pos.y)
        new_case = self.grid[pos.x + move[0]][pos.y + move[1]]
        old_case = self.g_hist[g]
        if new_case != g:
            self.g_hist[g] = new_case
        print(f"oc = {old_case} | nc = {new_case}")
        can_move = True
        if new_case == PAC:
            if not self.super:
                self.current_score -= 180 # additional point will be count as a win game
                self.game_over = True
                return
            can_move = False
        if can_move:
            self.pos.move(move, g)
            self.grid[old_pos[0]][old_pos[1]] = old_case
            self.grid[pos.x][pos.y] = g
            print(f"o {old_pos} n ({pos.x}, {pos.y})")

    def score(self) -> float:
        return self.current_score

    def ghost_available_actions(self, pos, speed):
        actions = [0]
        nogo = [WAL, GAT, RED, BLU, PIN, ORA]
        if self.super:
            nogo.append(PAC)
        print(nogo, pos.x, pos.y)
        print(self.grid[pos.x - 1][pos.y])
        print(self.grid[pos.x][pos.y + 1])
        print(self.grid[pos.x + 1][pos.y])
        print(self.grid[pos.x][pos.y - 1])
        if self.grid[pos.x - 1][pos.y] not in nogo:
            actions.append(Action.UP.value)
        if self.grid[pos.x][pos.y + 1] not in nogo:
            actions.append(Action.RIGHT.value)
        if self.grid[pos.x + 1][pos.y] not in nogo:
            actions.append(Action.DOWN.value)
        if self.grid[pos.x][pos.y - 1] not in nogo:
            actions.append(Action.LEFT.value)
        return np.array(actions)

    def available_actions_ids(self) -> np.ndarray:
        pos = self.pos.pac
        actions = [0]
        if self.grid[pos.x - 1][pos.y] not in [WAL, GAT]:
            actions.append(Action.UP.value)
        if self.grid[pos.x][pos.y + 1] not in [WAL, GAT]:
            actions.append(Action.RIGHT.value)
        if self.grid[pos.x + 1][pos.y] not in [WAL, GAT]:
            actions.append(Action.DOWN.value)
        if self.grid[pos.x][pos.y - 1] not in [WAL, GAT]:
            actions.append(Action.LEFT.value)
        return np.array(actions)

    def reset(self):
        self.grid = BASE_GRID
        self.pos = Position()
        self.speed = (0, 0)
        self.game_over = False
        self.current_score = 0.0
        self.super = False
        self.ghost_pt = 20
        self.count = COUNT
        self.g_island = [BLU, ORA, PIN]
        self.g_ingame = [RED]
        self.g_speed = {
            RED: (0, 0),
            BLU: (0, 0),
            PIN: (0, 0),
            ORA: (0, 0)
        }
        self.g_hist = {
            RED: EMP,
            BLU: EMP,
            PIN: EMP,
            ORA: EMP
        }

    def view(self):
        print(f'Score : {self.score()}')
        print(f'Game Over : {self.is_game_over()}')
        for row in self.grid:
            for col in row:
                terrain = terrain_dict[col]
                color = terrain["c"]
                symbol = terrain["s"]
                print(f"{color}{symbol}{Color.CLEAR.value}", end='')
            print()

    def copy(self):
        copy = Pacman()
        copy.grid = self.grid.copy()
        copy.game_over = self.game_over
        copy.current_score = self.current_score
        copy.pos = Position(copy=self.pos)
        copy.speed = self.speed
        copy.super = self.super
        copy.ghost_pt = self.ghost_pt
        copy.count = self.count
        copy.g_island = self.g_island
        copy.g_ingame = self.g_ingame
        copy.g_speed = dict(self.g_speed)
        copy.g_hist = dict(self.g_hist)

        return copy

    def get_grid(self):
        return self.grid

def main():
  pac = Pacman()
  pac.view()
  print(f"{Color.BLUE.value}#{Color.CLEAR.value}")


if __name__ == '__main__':
  main()