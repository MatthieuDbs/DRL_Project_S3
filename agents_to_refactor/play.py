import time
import os 
import numpy as np 
import tensorflow as tf
from env import DeepSingleAgentEnv, LineWorld
from ppo_multi_actor import model_prediction

def play(policy):
    env = LineWorld(10)
    env.view()
    time.sleep(1)
    while not env.is_game_over():
        
        s = env.state_description()        
        aa = env.available_actions_ids()
        mask = np.zeros((env.max_action_count(),))
        mask[aa] = 1.0
        pi_s_pred, v_s_pred = model_prediction(policy, [np.array([s]), np.array([mask])])        
        chosen_action = tf.squeeze(tf.random.categorical(tf.math.log(pi_s_pred), 1))
        
        #print(chosen_action)
        env.act_with_action_id(chosen_action)
        
        #clear_output(wait=True)
        os.system('cls')
        env.view()
        time.sleep(1)
        os.system('cls')


if __name__ == '__main__':

    policy = tf.keras.models.load_model("LineWord/ppo_multi_actor_model")
    play(policy)