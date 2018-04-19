import gym
import copy
import time
import numpy as np
# import multiprocessing as mp

GameName = ['CartPole-v0', 'MountainCar-v0', 'MsPacman-ram-v0']

SIGMA = 0.05
POPSIZE = 20
MAXGEN = 5000

np.random.seed(0)


class Env(object):
    def __init__(self, name, max_step):
        self.name = name
        self.f = gym.make(name).unwrapped
        self.n_in = self.f.observation_space.shape[0]
        self.n_out = self.f.action_space.n

        self.max_step = max_step

    def evaluate(self, nn):
        s = self.f.reset()
        reward = 0
        for step in range(self.max_step):
            a = nn.get_action(s)
            s, r, done, _ = self.f.step(a)
            # modify the reward of MountainCar
            if self.name == 'MountainCar-v0' and s[0] > -0.1:
                r = 0
            reward += r
            if done:
                break
        return reward

    def show(self, nn, interval):
        while True:
            s = self.f.reset()
            for step in range(self.max_step):
                self.f.render()
                time.sleep(interval)
                a = nn.get_action(s)
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
                                     n_hide * n_out + n_out) * SIGMA

        self.v = np.zeros_like(self.layer)
        self.lr, self.mom = 0.05, 0.9

    def reshape(self):
        params, count = [], 0
        for shape in self.shape:
            ws, we = count, count + shape[1] * shape[0]
            bs, be = we, we + shape[1]
            params.append([self.layer[ws:we].reshape((shape[1], shape[0])),
                           self.layer[bs:be].reshape((shape[1], 1))])
            count += shape[0] * shape[1] + shape[1]
        return params

    def forward(self, state):
        nn = self.reshape()
        x = state.reshape(len(state), 1)
        x = np.tanh(np.dot(nn[0][0], x) + nn[0][1])
        x = np.tanh(np.dot(nn[1][0], x) + nn[1][1])
        x = np.dot(nn[2][0], x) + nn[2][1]
        return x

    def get_action(self, x):
        y = self.forward(x)
        return np.argmax(y)

    def modify_params(self, delta):
        self.layer += delta

    def update_params(self, grad):
        self.v = self.mom * self.v + (1 - self.mom) * grad
        self.layer += self.lr * self.v

    def save(self):
        np.save("nn.npy", [self.shape, self.layer])

    def load(self):
        [self.shape, self.layer] = np.load("nn.npy")


class ES(object):
    @staticmethod
    def mirror(n): return (-1) ** (n % 2)

    def __init__(self, popsize):
        self.popsize = popsize
        self.population = np.random.randint(1, 2 ** 32 - 1, size=int(self.popsize / 2)).repeat(2)

        rank = np.arange(1, self.popsize + 1)
        temp = np.maximum(0, np.log(self.popsize / 2 + 1) - np.log(rank))
        self.w = temp / temp.sum() - 1 / self.popsize

    def evolution(self, nn, env):
        reward = []
        for i in range(self.popsize):
            np.random.seed(self.population[i])
            nnn = copy.deepcopy(nn)
            noise = self.mirror(i) * SIGMA * np.random.randn(len(nn.layer))
            nnn.modify_params(noise)
            reward.append(env.evaluate(nnn))
        rank = np.argsort(reward)[::-1]

        update = np.zeros(len(nn.layer))
        for i, kid in enumerate(rank):
            np.random.seed(self.population[kid])
            update += self.mirror(kid) * self.w[i] * np.random.randn(len(nn.layer))
        gradients = update / (self.popsize * SIGMA)
        nn.update_params(gradients)

        return np.average(reward)


def learning():
    env = Env(GameName[1], 500)
    net = NeuralNetwork(env.n_in, 30, env.n_out)
    es = ES(POPSIZE)

    net_cr = None
    for gen in range(MAXGEN):
        ts = time.time()
        kids_ar = es.evolution(net, env)
        net_r = env.evaluate(net)
        net_cr = net_r if net_cr is None else 0.9 * net_cr + 0.1 * net_r
        te = time.time()
        print('Gen: ', gen,
              ' net_r: %.3f' % net_r,
              ' kids_ar: %.3f' % kids_ar,
              ' net_cr: %.3f' % net_cr,
              ' t: %.3f' % (te - ts))

    net.save()
    env.show(net, 0.01)


if __name__ == "__main__":
    learning()
