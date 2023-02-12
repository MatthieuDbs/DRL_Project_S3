import math
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import matplotlib.pyplot as plt
import numpy as np
import numpy.random
import tensorflow as tf
from tqdm import tqdm
from env import DeepSingleAgentEnv, LineWorld




@tf.function
def model_prediction(pi_and_v_model: tf.keras.models.Model,
                     model_inputs):
    return pi_and_v_model(model_inputs)


@tf.function
def training_step(pi_and_v_model: tf.keras.models.Model,
                  model_inputs,
                  a: int,
                  target: float,
                  pi_old: np.ndarray,
                  delta: float,
                  c1: float,
                  c2: float,
                  epochs: int,
                  opt):
    for _ in range(epochs):
        with tf.GradientTape() as tape:
            pi_s_pred, v_s_pred = pi_and_v_model(model_inputs)
            loss_vf = (target - v_s_pred[0][0]) ** 2
            loss_entropy = - tf.reduce_sum(pi_s_pred[0] * tf.math.log(pi_s_pred[0] + 0.000000001))
            r = pi_s_pred[0][a] / (pi_old[0][a] + 0.000000001)
            loss_policy_clipped = tf.minimum(r * delta,
                                             tf.clip_by_value(r, 1.0 - 0.2, 1 + 0.2) * delta)
            total_loss = -loss_policy_clipped + c1 * loss_vf - c2 * loss_entropy

        grads = tape.gradient(total_loss, pi_and_v_model.trainable_variables)
        opt.apply_gradients(zip(grads, pi_and_v_model.trainable_variables))





def ppo(env: DeepSingleAgentEnv, max_iter_count: int = 10000,
        gamma: float = 0.99,
        learning_rate: float = 3e-4,
        epochs: int = 5,
        c1: float = 1.0,
        c2: float = 0.01):
    pi_and_v_input_state_desc = tf.keras.layers.Input((env.state_dim(),))
    pi_and_v_input_mask = tf.keras.layers.Input((env.max_action_count(),))
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    hidden_tensor = pi_and_v_input_state_desc
    for _ in range(3):
        hidden_tensor = tf.keras.layers.Dense(128,
                                              activation=tf.keras.activations.tanh,
                                              use_bias=True
                                              )(hidden_tensor)

    output_pi_tensor = tf.keras.layers.Dense(env.max_action_count(),
                                             activation=tf.keras.activations.linear,
                                             use_bias=True
                                             )(hidden_tensor)

    output_v_tensor = tf.keras.layers.Dense(1,
                                            activation=tf.keras.activations.linear,
                                            use_bias=True
                                            )(hidden_tensor)

    output_pi_probs = tf.keras.layers.Softmax()(output_pi_tensor, pi_and_v_input_mask)

    pi_and_v_model = tf.keras.models.Model([pi_and_v_input_state_desc, pi_and_v_input_mask],
                                           [output_pi_probs, output_v_tensor])

    ema_score = 0.0
    ema_nb_steps = 0.0
    first_episode = True

    step = 0
    ema_score_progress = []
    ema_nb_steps_progress = []

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

        s = env.state_description()

        aa = env.available_actions_ids()

        mask = np.zeros((env.max_action_count(),))
        mask[aa] = 1.0

        pi_s_pred, v_s_pred = model_prediction(pi_and_v_model, [np.array([s]), np.array([mask])])

        allowed_pi_s = pi_s_pred[0].numpy()[aa]
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
        aa_p = env.available_actions_ids()

        mask_p = np.zeros((env.max_action_count(),))

        if len(aa_p) > 0:
            mask_p[aa_p] = 1.0

        ### TRAINING TIME !!!

        pi_s_p_pred, v_s_p_pred = model_prediction(pi_and_v_model, [np.array([s_p]), np.array([mask_p])])

        target = r if env.is_game_over() else r + gamma * v_s_p_pred[0][0]
        delta = target - tf.constant(v_s_pred[0][0])  # for now it's At = Advantage of playing action a
        pi_old = tf.constant(pi_s_pred)

        training_step(pi_and_v_model, [np.array([s]), np.array([mask])], a, target, pi_old, delta, c1, c2, epochs, opt)

        step += 1
    return pi_and_v_model, ema_score_progress, ema_nb_steps_progress


pi_and_v_model, scores, steps = ppo(LineWorld(9), max_iter_count=10000)
print(pi_and_v_model.weights)
plt.plot(scores)
plt.show()
plt.plot(steps)
plt.show()