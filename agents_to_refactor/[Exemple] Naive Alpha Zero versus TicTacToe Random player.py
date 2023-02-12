import math
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import matplotlib.pyplot as plt
import numpy as np
import numpy.random
import tensorflow as tf
from tqdm import tqdm


class ExpertApprenticeSingleAgentEnv:
    def state_id(self) -> int:
        pass

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

    def clone(self) -> 'ExpertApprenticeSingleAgentEnv':
        pass


class TTTVsRandom(ExpertApprenticeSingleAgentEnv):
    def __init__(self):
        self.grid = np.zeros((3, 3))
        self.game_over = False
        self.aa = [i for i in range(9)]
        self.current_score = 0.0
        self.id = 0

    def state_id(self) -> int:
        return self.id

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
        self.id += 3 ** action_id * 2
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

    def clone(self) -> 'ExpertApprenticeSingleAgentEnv':
        cloned_env = TTTVsRandom()
        cloned_env.grid = np.copy(self.grid)
        cloned_env.aa = self.aa.copy()
        cloned_env.current_score = self.current_score
        cloned_env.id = self.id
        return cloned_env


def choose_action_with_neural_mcts(env: ExpertApprenticeSingleAgentEnv,
                                   model: tf.keras.Model,
                                   iterations_count: int = 200) -> (int, np.ndarray):
    tree = {}

    q_min = 0.0
    q_max = 1.0

    root = env.state_id()

    tree[root] = {}
    s = env.state_description()

    aa = env.available_actions_ids()

    mask = np.zeros((env.max_action_count(),))
    if not env.is_game_over():
        mask[aa] = 1.0

    pi_s, v_s = model([np.array([s]), np.array([mask])])
    for a in env.available_actions_ids():
        tree[root][a] = {
            'mean_score': 0.0,
            'n_selections': 0,
            'n_considerations': 0,
            'pi_s_a': pi_s[0][a],
            'v_s': v_s[0][0]
        }

    for _ in range(iterations_count):
        cloned_env = env.clone()
        current_node = env.state_id()

        nodes_and_chosen_actions = []

        # SELECT
        while not cloned_env.is_game_over() and not any(filter(lambda a_stats: a_stats['n_selections'] == 0,
                                                               tree[current_node].values())):
            best_a = None
            best_a_score = None
            for a, a_stats in tree[current_node].items():

                q_normalized = (a_stats['mean_score'] - q_min) / (q_max - q_min)
                # UCB-1 Criterium
                score = q_normalized + math.sqrt(2) * a_stats['pi_s_a'] * math.sqrt(
                    math.log(a_stats['n_considerations']) /
                    a_stats['n_selections'])
                if best_a_score is None or score > best_a_score:
                    best_a = a
                    best_a_score = score

            nodes_and_chosen_actions.append((current_node, best_a))
            cloned_env.act_with_action_id(best_a)
            current_node = cloned_env.state_id()

            if current_node not in tree:
                tree[current_node] = {}
                s = cloned_env.state_description()

                aa = cloned_env.available_actions_ids()

                mask = np.zeros((cloned_env.max_action_count(),))
                if not cloned_env.is_game_over():
                    mask[aa] = 1.0

                pi_s, v_s = model([np.array([s]), np.array([mask])])
                for a in cloned_env.available_actions_ids():
                    tree[current_node][a] = {
                        'mean_score': 0.0,
                        'n_selections': 0,
                        'n_considerations': 0,
                        'pi_s_a': pi_s[0][a],
                        'v_s': v_s[0][0]
                    }

        # EXPAND
        if not cloned_env.is_game_over():
            # Choose an action which has never been played
            a = np.random.choice(list(map(lambda action_and_stats: action_and_stats[0],
                                          filter(lambda action_and_stats: action_and_stats[1]['n_selections'] == 0,
                                                 tree[current_node].items()))))

            nodes_and_chosen_actions.append((current_node, a))
            cloned_env.act_with_action_id(a)

            current_node = cloned_env.state_id()

            if current_node not in tree:
                tree[current_node] = {}
                s = cloned_env.state_description()

                aa = cloned_env.available_actions_ids()

                mask = np.zeros((cloned_env.max_action_count(),))
                if not cloned_env.is_game_over():
                    mask[aa] = 1.0

                pi_s, v_s = model([np.array([s]), np.array([mask])])
                for a in cloned_env.available_actions_ids():
                    tree[current_node][a] = {
                        'mean_score': 0.0,
                        'n_selections': 0,
                        'n_considerations': 0,
                        'pi_s_a': pi_s[0][a],
                        'v_s': v_s[0][0]
                    }

        # EVALUATE
        score = v_s[0][0] if not cloned_env.is_game_over() else cloned_env.score()

        # BACKPROP
        for (node, chosen_action) in nodes_and_chosen_actions:
            for a in tree[node].keys():
                tree[node][a]['n_considerations'] += 1

            tree[node][chosen_action]['mean_score'] = (
                    (tree[node][chosen_action]['mean_score'] * tree[node][chosen_action]['n_selections'] + score) /
                    (1 + tree[node][chosen_action]['n_selections'])
            )
            tree[node][chosen_action]['n_selections'] += 1
            q_min = np.minimum(q_min, tree[node][chosen_action]['mean_score'])
            q_max = np.maximum(q_max, tree[node][chosen_action]['mean_score'])


    actions = []
    selection_count_per_action = []
    selection_count_sum = 0.0
    for (a, a_stats) in tree[root].items():
        actions.append(a)
        selection_count_per_action.append(a_stats['n_selections'] * 1.0)
        selection_count_sum += a_stats['n_selections'] * 1.0

    selection_count_per_action = np.array(selection_count_per_action)
    selection_count_per_action /= selection_count_sum

    chosen_action = np.random.choice(actions, p=selection_count_per_action)

    actions_weights = np.zeros((env.max_action_count(),))
    actions_weights[actions] = selection_count_per_action

    return chosen_action, actions_weights


def naive_alpha_zero(env: ExpertApprenticeSingleAgentEnv,
                     model: tf.keras.Model,
                     games_count: int,
                     train_apprentice_every_n_games: int,
                     train_apprentice_epochs: int):
    total_score = 0.0
    wins = 0
    losses = 0
    draws = 0
    X_state_description = []
    X_mask = []
    Y_pi = []
    Y_v = []

    total_score_history = []
    ema_total_score_current_value = 0.0
    ema_total_score_history = []


    opt = tf.keras.optimizers.Adam()

    for game_id in tqdm(range(games_count)):
        env.reset()

        game_steps_count = 0
        while not env.is_game_over():
            chosen_action, target_pi = choose_action_with_neural_mcts(env, model, 30)

            X_state_description.append(env.state_description())

            aa = env.available_actions_ids()

            mask = np.zeros((env.max_action_count(),))
            mask[aa] = 1.0
            X_mask.append(mask)

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

        if (game_id + 1) % train_apprentice_every_n_games == 0:
            #TRAINING !!!
            X_train = [tf.constant(X_state_description, dtype=tf.float32),
                       tf.constant(X_mask, dtype=tf.float32)]
            Y_pi_true = tf.constant(Y_pi, dtype=tf.float32)
            Y_v_true = tf.constant(Y_v, dtype=tf.float32)

            for ep in range(train_apprentice_epochs):
                with tf.GradientTape() as tape:
                    Y_pi_pred, Y_v_pred = model(X_train)
                    pi_loss = -tf.reduce_sum(Y_pi_true * tf.math.log(Y_pi_pred + 0.0000000001), axis=-1)
                    v_loss = (Y_v_true - tf.squeeze(Y_v_pred)) ** 2
                    loss = tf.reduce_mean(pi_loss + v_loss)

                grads = tape.gradient(loss, model.trainable_variables)
                opt.apply_gradients(zip(grads, model.trainable_variables))

            print(f'Expert (MCTS) mean score : {total_score / train_apprentice_every_n_games}')
            print(f'Expert (MCTS) outcomes % : wins: {wins / train_apprentice_every_n_games}, losses: {losses / train_apprentice_every_n_games}, draws: {draws / train_apprentice_every_n_games}')

            total_score_history.append(total_score / train_apprentice_every_n_games)
            ema_total_score_current_value = (1 - 0.9) * total_score / train_apprentice_every_n_games + 0.9 * ema_total_score_current_value
            ema_total_score_history.append(ema_total_score_current_value / (1 - 0.9 ** (len(ema_total_score_history) + 1)))

            plt.plot(np.arange(len(total_score_history)), total_score_history)
            plt.plot(np.arange(len(total_score_history)), ema_total_score_history)
            plt.show()

            total_score = 0.0
            wins = 0.0
            losses = 0.0
            draws = 0.0
            X_state_description.clear()
            X_mask.clear()
            Y_pi.clear()
            Y_v.clear()

    return model


def create_model(env: ExpertApprenticeSingleAgentEnv):
    pi_input_state_desc = tf.keras.layers.Input((env.state_dim(),))
    pi_input_mask = tf.keras.layers.Input((env.max_action_count(),))

    hidden_tensor = pi_input_state_desc
    for _ in range(2):
        hidden_tensor = tf.keras.layers.Dense(128,
                                              activation=tf.keras.activations.tanh,
                                              use_bias=True
                                              )(hidden_tensor)

    output_tensor = tf.keras.layers.Dense(env.max_action_count(),
                                          activation=tf.keras.activations.linear,
                                          use_bias=True
                                          )(hidden_tensor)

    output_probs = tf.keras.layers.Softmax()(output_tensor, pi_input_mask)

    output_value = tf.keras.layers.Dense(1,
                                         activation=tf.keras.activations.linear,
                                         use_bias=True
                                         )(hidden_tensor)

    model = tf.keras.models.Model([pi_input_state_desc, pi_input_mask], [output_probs, output_value])

    return model


def train_model(env: ExpertApprenticeSingleAgentEnv, X_train, Y_train, epochs: int = 10000) -> tf.keras.models.Model:
    # create model
    pi_input_state_desc = tf.keras.layers.Input((env.state_dim(),))
    pi_input_mask = tf.keras.layers.Input((env.max_action_count(),))

    hidden_tensor = pi_input_state_desc
    for _ in range(3):
        hidden_tensor = tf.keras.layers.Dense(128,
                                              activation=tf.keras.activations.tanh,
                                              use_bias=True
                                              )(hidden_tensor)

    output_tensor = tf.keras.layers.Dense(env.max_action_count(),
                                          activation=tf.keras.activations.linear,
                                          use_bias=True
                                          )(hidden_tensor)

    output_probs = tf.keras.layers.Softmax()(output_tensor, pi_input_mask)

    pi = tf.keras.models.Model([pi_input_state_desc, pi_input_mask], output_probs)

    opt = tf.keras.optimizers.Adam()

    for ep in range(epochs):
        with tf.GradientTape() as tape:
            Y_pred = pi(X_train)
            loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(Y_train, Y_pred))

        if ep % 1000 == 0:
            print(loss)
        grads = tape.gradient(loss, pi.trainable_variables)
        opt.apply_gradients(zip(grads, pi.trainable_variables))

    return pi


def run_ttt_for_n_games(training_games_count: int, games_count: int):
    env = TTTVsRandom()

    model = create_model(env)

    model = naive_alpha_zero(env, model, training_games_count, 30, 5)

    total_score = 0.0
    wins = 0
    losses = 0
    draws = 0
    for _ in tqdm(range(games_count)):
        env.reset()

        while not env.is_game_over():
            s = env.state_description()

            aa = env.available_actions_ids()

            mask = np.zeros((env.max_action_count(),))
            mask[aa] = 1.0

            pi_s, _ = model([np.array([s]), np.array([mask])])
            pi_s = np.array(pi_s[0])
            allowed_pi_s = pi_s[aa]
            sum_allowed_pi_s = np.sum(allowed_pi_s)
            if sum_allowed_pi_s == 0.0:
                probs = np.ones((len(aa),)) * 1.0 / (len(aa))
            else:
                probs = allowed_pi_s / sum_allowed_pi_s

            chosen_action = np.random.choice(aa, p=probs)

            env.act_with_action_id(chosen_action)

        if env.score() > 0:
            wins += 1
        elif env.score() < 0:
            losses += 1
        else:
            draws += 1
        total_score += env.score()
    print(f'Apprentice (Neural Net) mean score : {total_score / games_count}')
    print(
        f'Apprentice (Neural Net) outcomes % : wins: {wins / games_count}, losses: {losses / games_count}, draws: {draws / games_count}')
    return total_score / games_count


if __name__ == "__main__":
    run_ttt_for_n_games(1000, 1000)
