import numpy as np

np.random.seed(0)

GloOpi = 1e20
MaxGeneration = 1000
PopSize = 20
CR, F = 0.5, 0.5
LB, UB = -10., 10.
D = 50
nn, nnn = [], []
count = 0

class Individual:
    def __init__(self):
        self.gene = np.zeros(D)
        self.fitness = 0

    def initialize(self):
        global D, LB, UB
        self.gene = LB + np.random.random(D)*(UB - LB)
        self.func()

    def func(self):
        global D
        self.fitness = 0
        for i in range(D):
            self.fitness += self.gene[i]**2
        global count
        count += 1
        f.write('evals: ' + str(count) + ' fitness: ' + str("%.6f" % GloOpi) + '\n')


def initialize():
    global PopSize, nn
    for i in range(PopSize):
        nn.append(Individual())
        nn[i].initialize()


def production():
    global PopSize, nn, GloOpi

    for i in range(PopSize):
        nnn.append(Individual())
    #主过程
    for i in range(PopSize):
        ind = np.random.choice(np.arange(PopSize), 3, replace=False)
        j = np.random.randint(D)

        trial = Individual()
        for k in range(D):#交叉变异
            if np.random.random() < CR or k == D - 1:
                trial.gene[j] = nn[ind[0]].gene[j] + F * (nn[ind[1]].gene[j] - nn[ind[2]].gene[j])
            else:
                trial.gene[j] = nn[i].gene[j]
            j = (j + 1) % D
        trial.func()
        #自然选择
        GloOpi = min(GloOpi, trial.fitness)
        if trial.fitness <= nn[i].fitness:
            nnn[i] = trial
        else:
            nnn[i] = nn[i]
    nn = nnn


def main():
    initialize()
    for i in range(MaxGeneration):
        production()


if __name__ == "__main__":
    f = open('result.txt', 'w')
    main()
    f.close()
