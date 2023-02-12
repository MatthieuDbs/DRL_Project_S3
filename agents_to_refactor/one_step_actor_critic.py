

import math
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import matplotlib.pyplot as plt
import numpy as np
import numpy.random
import tensorflow as tf
from tqdm import tqdm
from env import DeepSingleAgentEnv, LineWorld



def one_step_actor_critic(env: DeepSingleAgentEnv, max_iter_count: int = 10000,
                          gamma: float = 0.99,
                          alpha_pi: float = 0.01,
                          alpha_v: float = 0.05):
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

    v = tf.keras.models.Sequential()
    for _ in range(3):
        v.add(tf.keras.layers.Dense(128,
                                    activation=tf.keras.activations.tanh,
                                    use_bias=True
                                    ))
    v.add(tf.keras.layers.Dense(1,
                                activation=tf.keras.activations.linear,
                                use_bias=True
                                ))

    ema_score = 0.0
    ema_nb_steps = 0.0
    first_episode = True

    step = 0
    ema_score_progress = []
    ema_nb_steps_progress = []

    I = 1.0

    for _ in tqdm(range(max_iter_count)):
        if env.is_game_over():
            if first_episode:
                ema_score = env.score()
                ema_nb_steps = step
                first_episode = False
            else:
                ema_score = (1 - 0.99) * env.score() + 0.99 * ema_score
                ema_nb_steps = (1 - 0.99) * step + 0.99 * ema_nb_steps
                ema_score_progress.append(ema_score)
                ema_nb_steps_progress.append(ema_nb_steps)

            env.reset()
            step = 0
            I = 1.0

        s = env.state_description()

        aa = env.available_actions_ids()

        mask = np.zeros((env.max_action_count(),))
        mask[aa] = 1.0

        pi_s = pi([np.array([s]), np.array([mask])])[0].numpy()
        allowed_pi_s = pi_s[aa]
        sum_allowed_pi_s = np.sum(allowed_pi_s)
        if sum_allowed_pi_s == 0.0:
            probs = np.ones((len(aa),)) * 1.0 / (len(aa))
        else:
            probs = allowed_pi_s / sum_allowed_pi_s

        a = np.random.choice(aa, p=probs)

        old_score = env.score()
        env.act_with_action_id(a)
        new_score = env.score()
        r = new_score - old_score

        s_p = env.state_description()

        ### TRAINING TIME !!!

        with tf.GradientTape() as tape_v:
            v_s_pred = v(np.array([s]))[0][0]

        target = r if env.is_game_over() else r + gamma * v(np.array([s_p]))[0][0]
        delta = target - tf.constant(v_s_pred)

        grad_v_s_pred = tape_v.gradient(v_s_pred, v.trainable_variables)
        for (var, grad) in zip(v.trainable_variables, grad_v_s_pred):
            if grad is not None:
                var.assign_add(alpha_v * delta * grad)

        with tf.GradientTape() as tape_pi:
            pi_s_a_t = pi([
                np.array([s]),
                np.array([mask])
            ])[0][a]
            log_pi_s_a_t = tf.math.log(pi_s_a_t)

        grads = tape_pi.gradient(log_pi_s_a_t, pi.trainable_variables)

        for (var, grad) in zip(pi.trainable_variables, grads):
            if grad is not None:
                var.assign_add(alpha_pi * I * delta * grad)

        I = I * gamma
        step += 1
    return pi, v, ema_score_progress, ema_nb_steps_progress


pi, v, scores, steps = one_step_actor_critic(LineWorld(10), max_iter_count=5000)
print(pi.weights)
plt.plot(scores)
plt.show()
plt.plot(steps)
plt.show()
