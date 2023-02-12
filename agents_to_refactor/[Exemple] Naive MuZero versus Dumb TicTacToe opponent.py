import math
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import matplotlib.pyplot as plt
import numpy as np
import numpy.random
import tensorflow as tf
from tqdm import tqdm


class MuZeroAgentEnv:

    def max_action_count(self) -> int:
        pass

    def state_description(self) -> np.ndarray:
        pass

    def state_dim(self) -> int:
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

    def clone(self) -> 'MuZeroAgentEnv':
        pass


class TTTVsDumbPlayer(MuZeroAgentEnv):
    def __init__(self):
        self.grid = np.zeros((3, 3))
        self.game_over = False
        self.aa = [i for i in range(9)]
        self.current_score = 0.0

    def max_action_count(self) -> int:
        return 9

    def state_description(self) -> np.ndarray:
        return tf.keras.utils.to_categorical(self.grid, 3).flatten()

    def state_dim(self) -> int:
        return 27

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

        # action_id = np.random.choice(self.aa)
        action_id = self.aa[0]
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

    def clone(self) -> 'MuZeroAgentEnv':
        cloned_env = TTTVsDumbPlayer()
        cloned_env.grid = np.copy(self.grid)
        cloned_env.aa = self.aa.copy()
        cloned_env.current_score = self.current_score
        return cloned_env


class Edge:
    def __init__(self, a: int, pi_a: float, node: 'Node'):
        self.a = a
        self.pi_a = pi_a
        self.parent_node = node
        self.child_node = None
        self.n = 0
        self.c = 0
        self.q = 0.0


class Node:
    def __init__(self, s: tf.Tensor, pi: tf.Tensor, v: float, max_actions_count: int):
        self.s = s
        self.v = v
        self.child_edges = [Edge(a, pi[a], self) for a in range(max_actions_count)]
        self.parent_edge = None


def full_neural_mcts(
        h: tf.keras.Model,
        g: tf.keras.Model,
        f: tf.keras.Model,
        original_env_observation: np.ndarray,
        original_action_mask: np.ndarray,
        max_action_count: int,
        iteration_count: int,
):
    s0 = h(tf.expand_dims(original_env_observation, 0))
    p, v = f(s0)

    root = Node(s0, p[0], v[0][0], max_action_count)

    q_min = 0.0
    q_max = 1.0

    for it in range(iteration_count):
        current_node = root

        # SELECT
        chosen_edge = None
        while chosen_edge is None:
            best_edge = None
            best_edge_score = None

            for edge in current_node.child_edges:
                if original_action_mask[edge.a] == 0 and current_node == root:
                    continue

                q_normalized = (edge.q - q_min) / (q_max - q_min)
                score = q_normalized + edge.pi_a * np.sqrt(edge.c) / (1 + edge.n) * (1.25 +
                                                                                     np.log(
                                                                                         (edge.c + 19652 + 1) / 19652
                                                                                     )
                                                                                     )
                if best_edge_score is None or score > best_edge_score:
                    best_edge = edge
                    best_edge_score = score

            if best_edge.child_node is None:
                chosen_edge = best_edge
            else:
                current_node = best_edge.child_node

        # EXPAND
        r, s = g([current_node.s, tf.expand_dims(tf.one_hot(chosen_edge.a, max_action_count), 0)])
        p, v = f(s)
        child_node = Node(s, p[0], v[0][0], max_action_count)
        chosen_edge.child_node = child_node
        child_node.parent_edge = chosen_edge

        # EVALUATE
        score = v[0][0]

        # BACKUP
        current_node = child_node
        while current_node.parent_edge is not None:
            edge = current_node.parent_edge
            edge.q = (edge.q * edge.n + score) / (1 + edge.q)
            q_min = np.minimum(q_min, edge.q)
            q_max = np.maximum(q_max, edge.q)

            edge.n += 1
            current_node = edge.parent_node
            for edge in current_node.child_edges:
                edge.c += 1

    pi = np.zeros((max_action_count,))
    for edge in root.child_edges:
        pi[edge.a] = edge.n / iteration_count

    mcts_chosen_action = np.random.choice(np.arange(max_action_count), p=pi)
    return mcts_chosen_action, pi


def naive_mu_zero(env: MuZeroAgentEnv,
                  h: tf.keras.Model,
                  g: tf.keras.Model,
                  f: tf.keras.Model,
                  games_count: int,
                  plot_stats_every_n_games: int):
    total_score = 0.0
    wins = 0
    losses = 0
    draws = 0
    X_state_description = []
    X_chosen_actions = []
    Y_pi = []
    Y_v = []

    total_score_history = []
    ema_total_score_current_value = 0.0
    ema_total_score_history = []

    opt = tf.keras.optimizers.Adam()
    all_trainable_variables = h.trainable_variables + g.trainable_variables + f.trainable_variables

    for game_id in tqdm(range(games_count)):
        env.reset()

        game_steps_count = 0
        while not env.is_game_over():
            aa = env.available_actions_ids()
            mask = np.zeros((env.max_action_count(),))
            mask[aa] = 1.0

            chosen_action, target_pi = full_neural_mcts(h, g, f, env.state_description(), mask,
                                                        env.max_action_count(),
                                                        30)

            X_state_description.append(env.state_description())
            X_chosen_actions.append(chosen_action)
            Y_pi.append(target_pi)

            env.act_with_action_id(chosen_action)
            game_steps_count += 1

        if env.score() > 0:
            wins += 1
        elif env.score() < 0:
            losses += 1
        else:
            draws += 1
        total_score += env.score()

        for step in range(game_steps_count):
            Y_v.append(env.score() * 0.99 ** (game_steps_count - 1 - step))

        # TRAINING

        with tf.GradientTape() as tape:
            s = h(tf.expand_dims(X_state_description[0], 0))
            pi_loss = 0.0
            v_loss = 0.0
            r_loss = 0.0

            for t in range(game_steps_count):
                p, v = f(s)

                pi_loss += - tf.reduce_sum(Y_pi[t] * tf.math.log(p[0] + 0.000000000001), axis=-1)
                v_loss += (Y_v[t] - v[0][0]) ** 2

                r, s = g([s, tf.expand_dims(tf.one_hot(X_chosen_actions[t], env.max_action_count()), 0)])

                r_loss += (r[0][0] - r[0][0])  # this is for silencing a warning for now

            loss = (pi_loss + v_loss + r_loss) / game_steps_count

        grads = tape.gradient(loss, all_trainable_variables)
        opt.apply_gradients(zip(grads, all_trainable_variables))

        X_state_description.clear()
        X_chosen_actions.clear()
        Y_pi.clear()
        Y_v.clear()

        if (game_id + 1) % plot_stats_every_n_games == 0:
            print(f'Expert (MCTS) mean score : {total_score / plot_stats_every_n_games}')
            print(
                f'Expert (MCTS) outcomes % : wins: {wins / plot_stats_every_n_games}, losses: {losses / plot_stats_every_n_games}, draws: {draws / plot_stats_every_n_games}')

            total_score_history.append(total_score / plot_stats_every_n_games)
            ema_total_score_current_value = (
                                                        1 - 0.9) * total_score / plot_stats_every_n_games + 0.9 * ema_total_score_current_value
            ema_total_score_history.append(
                ema_total_score_current_value / (1 - 0.9 ** (len(ema_total_score_history) + 1)))

            plt.plot(np.arange(len(total_score_history)), total_score_history)
            plt.plot(np.arange(len(total_score_history)), ema_total_score_history)
            plt.show()

            total_score = 0.0
            wins = 0.0
            losses = 0.0
            draws = 0.0


LATENT_STATE_DIM = 16


def create_models(env: MuZeroAgentEnv) -> (tf.keras.Model, tf.keras.Model, tf.keras.Model,):
    # CREATE OBSERVATION EMBEDDING MODEL
    input_o = tf.keras.layers.Input((env.state_dim(),))

    hidden_tensor = input_o
    for _ in range(1):
        hidden_tensor = tf.keras.layers.Dense(32,
                                              activation=tf.keras.activations.tanh,
                                              use_bias=True
                                              )(hidden_tensor)
    output_s = tf.keras.layers.Dense(LATENT_STATE_DIM,
                                     activation=tf.keras.activations.tanh,
                                     use_bias=True
                                     )(hidden_tensor)

    h = tf.keras.Model(input_o, output_s)

    # CREATE LATENT SPACE ENVIRONMENT DYNAMICS MODEL
    input_s = tf.keras.layers.Input((LATENT_STATE_DIM,))
    input_a = tf.keras.layers.Input((env.max_action_count(),))

    hidden_tensor = tf.keras.layers.Concatenate()([input_s, input_a])
    for _ in range(1):
        hidden_tensor = tf.keras.layers.Dense(32,
                                              activation=tf.keras.activations.tanh,
                                              use_bias=True
                                              )(hidden_tensor)
    output_r = tf.keras.layers.Dense(1,
                                     activation=tf.keras.activations.linear,
                                     use_bias=True
                                     )(hidden_tensor)
    output_s_p = tf.keras.layers.Dense(LATENT_STATE_DIM,
                                       activation=tf.keras.activations.tanh,
                                       use_bias=True
                                       )(hidden_tensor)
    g = tf.keras.Model([input_s, input_a], [output_r, output_s_p])

    # CREATE APPRENTICE
    input_s = tf.keras.layers.Input((LATENT_STATE_DIM,))

    hidden_tensor = input_s
    for _ in range(1):
        hidden_tensor = tf.keras.layers.Dense(32,
                                              activation=tf.keras.activations.tanh,
                                              use_bias=True
                                              )(hidden_tensor)

    output_tensor = tf.keras.layers.Dense(env.max_action_count(),
                                          activation=tf.keras.activations.linear,
                                          use_bias=True
                                          )(hidden_tensor)
    output_p = tf.keras.layers.Softmax()(output_tensor)

    output_v = tf.keras.layers.Dense(1,
                                     activation=tf.keras.activations.linear,
                                     use_bias=True
                                     )(hidden_tensor)

    f = tf.keras.models.Model(input_s, [output_p, output_v])

    return h, g, f


def run_ttt_for_n_games(training_games_count: int):
    env = TTTVsDumbPlayer()

    h, g, f = create_models(env)

    naive_mu_zero(env, h, g, f, training_games_count, 50)


if __name__ == "__main__":
    run_ttt_for_n_games(1000)
