import numpy as np


class BestSolution:

    def __init__(self):
        self.x, self.f = None, None
        self.evals = 0
        self.sol, self.fit = [], []

    def update(self, pop, popf):
        if self.f is None or self.f > popf:
            self.x, self.f = pop, popf
        self.evals += 1
        self.sol.append(self.x)
        self.fit.append(self.f)


class MAES:

    def __init__(self, prob):
        self.ans = BestSolution()
        self.prob = prob
        self.nn = prob.dim
        self.xx = np.random.random(self.nn) * (prob.ub - prob.lb) + prob.lb
        self.xmean = self.xx[:]
        self.sigma = 0.3

        self.lam = 4 + int(3 * np.log(self.nn))
        self.mu = int(self.lam/2)
        self.weights = [np.log(self.mu + 0.5) - np.log(i + 1) for i in range(self.mu)]
        self.weights = [w / sum(self.weights) for w in self.weights]
        self.mueff = sum(self.weights) ** 2 / sum(w ** 2 for w in self.weights)  # ???

        self.cs = (self.mueff + 2) / (self.nn + self.mueff + 5)
        self.c1 = 2 / ((self.nn + 1.3) ** 2 + self.mueff)
        self.cmu = min([1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((self.nn + 2) ** 2 + self.mueff)])
        self.damps = 1 + self.cs + 2 * max([0, ((self.mueff - 1)/self.nn) ** 0.5 - 1])  # ???

        self.ps = np.zeros(self.nn)
        self.M = np.eye(self.nn)

    def main_steps(self):
        # Sample
        newpop, z, d, fitvals = [], [], [], []
        for i in range(self.lam):
            zz = np.random.normal(0, 1, self.nn)
            dd = np.dot(self.M, zz)
            nn = self.xmean + self.sigma * dd
            z.append(zz), d.append(dd), newpop.append(nn)

        # sort and update mean
        fitvals = []
        for xx in newpop:
            fit = self.prob.evaluate(xx)
            self.ans.update(xx, fit)
            fitvals.append(fit)
        argx = np.argsort(fitvals)
        self.xmean = sum(self.weights[i] * newpop[argx[i]] for i in range(self.mu))  # Important

        # update evolution path
        zz = sum(self.weights[i] * z[argx[i]] for i in range(self.mu))
        c = (self.cs * (2 - self.cs) * self.mueff) ** 0.5
        self.ps -= self.cs * self.ps
        self.ps += c * zz

        # update matrix
        I = np.eye(self.nn, self.nn)
        part1 = I
        part2o = self.ps.reshape((self.nn, 1))
        part2t = self.ps.reshape((1, self.nn))
        part2 = self.c1 / 2 * (np.dot(part2o, part2t) - I)
        part3 = np.zeros((self.nn, self.nn))
        for i in range(self.mu):
            part3o = z[argx[i]].reshape((self.nn, 1))
            part3t = z[argx[i]].reshape((1, self.nn))
            part3 += self.weights[i] * np.dot(part3o, part3t)
        part3 = self.cmu / 2 * (part3 - I)
        self.M = np.dot(self.M, part1 + part2 + part3)

        # update step-size
        self.sigma *= np.exp(min(0.6, (self.cs / self.damps) * (sum(x ** 2 for x in self.ps) / self.nn - 1) / 2))

    def optimize(self, maxgens):
        for i in range(maxgens):
            print("maes: ", i)
            self.main_steps()

