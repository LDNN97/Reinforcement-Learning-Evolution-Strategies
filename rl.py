from env.maze import Maze
from ml.qlands import QLearningTable, SarsaTable


def main():
    # off policy
    # rl = QLearningTable(actions=list(range(env.n_actions)))
    # for episode in range(100):
    #     observation = env.reset()
    #     while True:
    #         action = rl.choose_action(str(observation))
    #         observation_, reward, done = env.step(action)
    #         rl.learn(str(observation), action, reward, str(observation_))
    #         observation = observation_
    #         if done:
    #             break

    # on policy
    rl = SarsaTable(actions=list(range(env.n_actions)))
    for episode in range(100):
        observation = env.reset()
        action = rl.choose_action(str(observation))
        while True:
            observation_, reward, done = env.step(action)
            action_ = rl.choose_action(str(observation_))
            rl.learn(str(observation), action, reward, str(observation_), action_)
            observation = observation_
            action = action_
            if done:
                break

    print('game over')
    env.destroy()


if __name__ == '__main__':
    env = Maze()
    env.after(1000, main)
    env.mainloop()
