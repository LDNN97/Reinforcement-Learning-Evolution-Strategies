import gym
import copy
import time
import numpy as np
# import multiprocessing as mp

GameName = ['CartPole-v0', 'MountainCar-v0', 'MsPacman-ram-v0']

SIGMA = 0.1
POPSIZE = 20
MAXGEN = 100


class Env(object):
    def __init__(self, name):
        self.f = gym.make(name).unwrapped
        self.n_in = self.f.observation_space.shape[0]
        self.n_out = self.f.action_space.n
        self.maxs = 1000

    def evaluate(self, nn):
        s = self.f.reset()
        reward = 0
        for i in range(self.maxs):
            a = nn.forward(s)
            s, r, done, _ = self.f.step(a)
            reward += r
            if done:
                break
        return reward

    def show(self, nn):
        while True:
            s = self.f.reset()
            for step in range(1000):
                self.f.render()
                time.sleep(0.05)
                a = nn.forward(s)
                s, _, done, _ = self.f.step(a)
                if done:
                    break


class NeuralNetwork(object):
    def __init__(self, n_in, n_hide, n_out):
        self.shape = []
        self.shape.append([n_in, n_hide])
        self.shape.append([n_hide, n_hide])
        self.shape.append([n_hide, n_out])
        self.layer = np.random.randn(n_in * n_hide + n_hide +
                                     n_hide * n_hide + n_hide +
                                     n_hide * n_out + n_out)

    def reshape(self):
        params, count = [], 0
        for shape in self.shape:
            ws, we = count, count + shape[1] * shape[0]
            bs, be = we, we + shape[1]
            params.append([self.layer[ws:we].reshape((shape[1], shape[0])),
                           self.layer[bs:be].reshape((shape[1], 1))])
            count += shape[0] * shape[1] + shape[1]
        return params

    @staticmethod
    def cal(params, x):
        return np.tanh(np.dot(params[0], x) + params[1])

    def forward(self, state):
        x = state.reshape(len(state), 1)
        nn = self.reshape()
        for i in range(3):
            x = NeuralNetwork.cal(nn[i], x)
        return np.argmax(x)

    def update_params(self, delta):
        self.layer += delta


class ES(object):
    def __init__(self, popsize):
        self.popsize = popsize
        self.population = np.random.randint(1, 2 ** 32 - 1, size=self.popsize)
        rank = np.arange(1, self.popsize + 1)
        temp = np.maximum(0, np.log(self.popsize / 2 + 1) - np.log(rank))
        self.w = temp / temp.sum() - 1 / self.popsize

    def evolution(self, nn, env):
        reward = []
        for i in range(self.popsize):
            np.random.seed(self.population[i])
            nnn = copy.deepcopy(nn)
            noise = SIGMA * np.random.randn(len(nn.layer))
            nnn.update_params(noise)
            reward.append(env.evaluate(nnn))
        rank = np.argsort(reward)[::-1]

        update = np.zeros(len(nn.layer))
        for i, kid in enumerate(rank):
            np.random.seed(self.population[kid])
            update += self.w[i] * np.random.randn(len(nn.layer))
        gradients = update / (self.popsize * SIGMA)
        nn.update_params(gradients)


def learning():
    env = Env(GameName[2])
    net = NeuralNetwork(env.n_in, 150, env.n_out)
    es = ES(POPSIZE)

    ar = None
    for gen in range(MAXGEN):
        ts = time.time()
        es.evolution(net, env)
        net_r = env.evaluate(net)
        ar = net_r if ar is None else 0.9 * ar + 0.1 * net_r
        te = time.time()
        print('Gen: ', gen, ' ar: %.3f' % ar, ' t: %.3f' % (te - ts))

    env.show(net)


if __name__ == "__main__":
    learning()
