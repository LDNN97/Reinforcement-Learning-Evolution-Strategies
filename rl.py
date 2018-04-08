from env.maze import Maze
from ml.qlearnging import QlearningTable


def main():
    rl = QlearningTable(actions=list(range(env.n_actions)))
    for episode in range(100):
        observation = env.reset()
        while True:
            action = rl.choose_action(str(observation))
            observation_, reward, done = env.step(action)
            rl.learn(str(observation), action, reward, str(observation_))
            observation = observation_
            if done:
                break

    print('game over')
    env.destroy()


if __name__ == '__main__':
    env = Maze()
    env.after(1000, main)
    env.mainloop()
