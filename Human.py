from envs import LineWorld, Pacman, GridWorld
import numpy as np
import os
from subprocess import call
 
# define clear function
def clear():
    # check and make call for specific operating system
    _ = call('clear' if os.name == 'posix' else 'cls')


ENV = Pacman


#TODO: metrics

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


def main():
    env = ENV()
    print("pacman:\n0 -> none\n1 -> up\n2 -> right\n3 -> down\n4 -> left")
    print("gridword:\n0 -> up\n1 -> right\n2 -> down\n3 -> left")
    input("press enter to start")
    while not env.is_game_over():
        clear()
        env.view()
        acts = env.available_actions_ids()
        print(acts)
        nogo = True
        while nogo:
            a = input("Enter an action from above: ")
            a = 0 if a == '' else int(a)
            nogo = False if a in acts else True
        env.act_with_action_id(a)


    print (f"final score: {env.score()}")


if __name__ == '__main__':
    main()