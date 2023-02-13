import math
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import matplotlib.pyplot as plt
import numpy as np
import numpy.random
import tensorflow as tf
from tqdm import tqdm

from envs import LineWorld, TTTVsRandom, Pacman, GridWorld

class NeuralMcts():
    def __init__(self, env, model: tf.keras.Model, it_count: int = 200):
        self.env = env
        self.model = model
        self.it_count = it_count

    def play(self, environment = None):
        env = self.env if not environment else environment

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

        pi_s, v_s = self.model([np.array([s]), np.array([mask])])
        for a in env.available_actions_ids():
            tree[root][a] = {
                'mean_score': 0.0,
                'n_selections': 0,
                'n_considerations': 0,
                'pi_s_a': pi_s[0][a],
                'v_s': v_s[0][0]
            }

        for _ in range(self.it_count):
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

                    pi_s, v_s = self.model([np.array([s]), np.array([mask])])
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

                    pi_s, v_s = self.model([np.array([s]), np.array([mask])])
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

class AlphaZero():
    def __init__(self, env, g_count, train_n_game, train_epoch, model = None, load: bool = False, filename = "./AlphaZero.model"):
        self.env = env
        if model:
            self.model = model
        else:
            if load:
                self.load(filename)
            else:
                self.create_model(env)
        self.g_count = g_count
        self.train_n_game = train_n_game
        self.train_epoch = train_epoch


    def create_model(self, env):
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

        self.model = tf.keras.models.Model([pi_input_state_desc, pi_input_mask], [output_probs, output_value])

        return self.model

    def train(self, deep = 100):
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

        for game_id in tqdm(range(self.g_count)):
            self.env.reset()

            game_steps_count = 0
            tdeep = deep
            while not self.env.is_game_over() and tdeep > 0:
                # self.env.view()
                chosen_action, target_pi = choose_action_with_neural_mcts(self.env, self.model, 30)

                X_state_description.append(self.env.state_description())

                aa = self.env.available_actions_ids()

                mask = np.zeros((self.env.max_action_count(),))
                mask[aa] = 1.0
                X_mask.append(mask)

                Y_pi.append(target_pi)

                self.env.act_with_action_id(chosen_action)
                game_steps_count += 1
                tdeep -= 1

            if self.env.score() > 0:
                wins += 1
            elif self.env.score() < 0:
                losses += 1
            else:
                draws += 1
            total_score += self.env.score()

            for step in range(game_steps_count):
                Y_v.append(self.env.score() * 0.99 ** (game_steps_count - 1 - step))

            if (game_id + 1) % self.train_n_game == 0:
                #TRAINING !!!
                X_train = [tf.constant(X_state_description, dtype=tf.float32),
                        tf.constant(X_mask, dtype=tf.float32)]
                Y_pi_true = tf.constant(Y_pi, dtype=tf.float32)
                Y_v_true = tf.constant(Y_v, dtype=tf.float32)

                for ep in range(self.train_epoch):
                    with tf.GradientTape() as tape:
                        Y_pi_pred, Y_v_pred = self.model(X_train)
                        pi_loss = -tf.reduce_sum(Y_pi_true * tf.math.log(Y_pi_pred + 0.0000000001), axis=-1)
                        v_loss = (Y_v_true - tf.squeeze(Y_v_pred)) ** 2
                        loss = tf.reduce_mean(pi_loss + v_loss)

                    grads = tape.gradient(loss, self.model.trainable_variables)
                    opt.apply_gradients(zip(grads, self.model.trainable_variables))

                print(f'Expert (MCTS) mean score : {total_score / self.train_n_game}')
                print(f'Expert (MCTS) outcomes % : wins: {wins / self.train_n_game}, losses: {losses / self.train_n_game}, draws: {draws / self.train_n_game}')

                total_score_history.append(total_score / self.train_n_game)
                ema_total_score_current_value = (1 - 0.9) * total_score / self.train_n_game + 0.9 * ema_total_score_current_value
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

        return self.model

    def save(self, filename = './AlphaZero.model'):
        self.model.save(filename)

    def load(self, filename = './AlphaZero.model'):
        self.model = tf.keras.models.load_model(filename)

def choose_action_with_neural_mcts(env,
                                   model: tf.keras.Model,
                                   iterations_count: int = 200, deep:int = 200):
    tree = {}
    globcount = 0

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
        # print (globcount)
        cloned_env = env.clone()
        current_node = env.state_id()

        nodes_and_chosen_actions = []
        tdeep = deep

        # SELECT
        while not cloned_env.is_game_over() and not any(filter(lambda a_stats: a_stats['n_selections'] == 0,
                                                               tree[current_node].values())) and tdeep > 0:
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
            tdeep -=1
            globcount += 1
            

        # EXPAND
        if not cloned_env.is_game_over() and tdeep > 0:
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
        score = cloned_env.score()

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
            globcount += 1


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


def naive_alpha_zero(env,
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


def create_model(env):
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


def train_model(env, X_train, Y_train, epochs: int = 10000) -> tf.keras.models.Model:
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
    env = Pacman()

    alpha0 = AlphaZero(env, training_games_count, 30, 5)

    alpha0.train()

    alpha0.save()

    model = alpha0.model

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
    run_ttt_for_n_games(100, 100)
