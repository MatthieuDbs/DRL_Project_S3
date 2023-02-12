import math
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from envs import LineWorld, TTTVsRandom, Pacman, GridWorld
import numpy as np

from libosmo import print_debug

ENV = Pacman


#TODO: metrics


class DoubleQNetwork():
    def __init__(self, env, max: int = 10000, g: float = 0.99, a: float = 0.1, e: float = 0.2, load: bool = False, filename = "./dqn.model"):
        self.env = env
        self.max = max
        self.g = g
        self.a = a
        self.e = e
        self.q = None
        if load:
            self.load(filename)

    def train(self):
        self.q = tf.keras.models.Sequential()
        self.q.add(tf.keras.layers.Dense(self.env.max_action_count(),
                                    activation=tf.keras.activations.linear,
                                    use_bias=True
                                    ))

        ema_score = 0.0
        ema_nb_steps = 0.0
        first_episode = True

        step = 0
        ema_score_progress = []
        ema_nb_steps_progress = []

        opt = tf.keras.optimizers.SGD(self.a)

        for _ in tqdm(range(self.max)):
            if self.env.is_game_over():
                if first_episode:
                    ema_score = self.env.score()
                    ema_nb_steps = step
                    first_episode = False
                else:
                    ema_score = (1 - 0.9) * self.env.score() + 0.9 * ema_score
                    ema_nb_steps = (1 - 0.9) * step + 0.9 * ema_nb_steps
                    ema_score_progress.append(ema_score)
                    ema_nb_steps_progress.append(ema_nb_steps)

                self.env.reset()
                step = 0

            s = self.env.state_description()
            aa = self.env.available_actions_ids()

            q_pred = self.q(np.array([s]))[0]
            if np.random.random() < self.e:
                a = np.random.choice(aa)
            else:
                print_debug("test", aa)
                print_debug(q_pred.numpy()) 
                print_debug(q_pred.numpy()[aa], np.argmax(q_pred.numpy()[aa]))
                a = aa[np.argmax(q_pred.numpy()[aa])]

            old_score = self.env.score()
            self.env.act_with_action_id(a)
            new_score = self.env.score()
            r = new_score - old_score

            s_p = self.env.state_description()
            aa_p = self.env.available_actions_ids()

            if self.env.is_game_over():
                y = r
            else:
                q_pred_p = self.q(np.array([s_p]))[0]
                print_debug(q_pred_p.numpy())
                max_q_s_p = np.max(q_pred_p.numpy()[aa_p])
                y = r + self.g * max_q_s_p

            with tf.GradientTape() as tape:
                q_s_a = self.q(np.array([s]))[0][a]
                loss = tf.reduce_mean((y - q_s_a) ** 2)

            grads = tape.gradient(loss, self.q.trainable_variables)
            opt.apply_gradients(zip(grads, self.q.trainable_variables))

            step += 1
        return self.q, ema_score_progress, ema_nb_steps_progress

    def save(self, filename = "./dqn.model"):
        self.q.save(filename)

    def load(self, filename = "./dqn.model"):
        self.q = tf.keras.models.load_model(filename)
        


agent = DoubleQNetwork(ENV(), e=0.1, max=10000)
q, scores, steps = agent.train()
agent.save()
print(q.weights)
plt.plot(scores)
plt.show()
plt.plot(steps)
plt.show()
