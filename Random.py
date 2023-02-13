from envs import LineWorld, TTTVsRandom, Pacman, GridWorld
import numpy as np

ENV = GridWorld


#TODO: metrics

def main():
    env = ENV()
    avg_step = 0
    avg_score = 0
    it = 0
    max_game = 1000 
    while it < max_game:
        step = 0
        while not env.is_game_over():
            step += 1
            env.view()
            acts = env.available_actions_ids()
            # print(acts)
            a = np.random.choice(acts)
            env.act_with_action_id(a)
            print(a)
        it +=1
        avg_score += env.score()
        avg_step += step
        env.reset()

    avg_step = avg_step / max_game
    avg_score = avg_score / max_game
    print(f"avg_score: {avg_score}, avg_step: {avg_step}")
    print (f"step: {step}")



if __name__ == '__main__':
    main()