from envs import LineWorld, TTTVsRandom, Pacman, GridWorld
import numpy as np

ENV = GridWorld


#TODO: metrics

def main():
    env = ENV()
    while not env.is_game_over():
        env.view()
        acts = env.available_actions_ids()
        # print(acts)
        a = np.random.choice(acts)
        env.act_with_action_id(a)
        print(a)

    env.view()



if __name__ == '__main__':
    main()