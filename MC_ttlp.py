from envs import LineWorld, TTTVsRandom, Pacman, GridWorld
import numpy as np




class MontecarloRR():
    def __init__(self, deep: int = 1000):
        self.deep = deep

    def play(self, env):
        env.view()
        envcopy = env.copy()
        actions = env.available_actions_ids()
        score = {a:0.0 for a in actions}

        for act in actions:
            envcopy = env.copy()
            envcopy.act_with_action_id(act)
            for _ in range(0, self.deep):
                envcopy2 = envcopy.copy()
                while not envcopy2.is_game_over():
                    nact = np.random.choice(envcopy2.available_actions_ids())
                    envcopy2.act_with_action_id(nact)
                score[act] += envcopy2.score()
            score[f"avg_{act}"] = score[act] / self.deep

        a = max(score, key=score.get)
        env.act_with_action_id(a)

# class MontecarloTTLP():
#     def __init__(self, deep: int = 1000):
#         self.deep = deep

#     def play(self, ttt: TTTVsRandom):
#         ttt.view()
#         tttcopy = ttt.copy()
#         actions = ttt.available_actions_ids()
#         score = {a:0.0 for a in actions}

#         for act in actions:
#             tttcopy = ttt.copy()
#             tttcopy.act_with_action_id(act)
#             for _ in range(0, self.deep):
#                 tttcopy2 = tttcopy.copy()
#                 while not tttcopy2.is_game_over():
#                     nact = random.choice(tttcopy2.available_actions_ids())
#                     tttcopy2.act_with_action_id(nact)
#                 score[act] += tttcopy2.score()
#             score[f"avg_{act}"] = score[act] / self.deep

#         a = max(score, key=score.get)
#         ttt.act_with_action_id(a)

ENV = GridWorld

#TODO: metrics

def main():
    env = ENV()
    agent = MontecarloRR()
    score = 0
    for i in range(50):
        print(f'game {i}:')
        while not env.is_game_over():
            agent.play(env)
        score += env.score()
        env.view()
        env.reset()

    print(score)


if __name__ == '__main__':
    main()