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


class CMAES:

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

        self.cc = (4 + self.mueff / self.nn) / (self.nn + 4 + 2 * self.mueff / self.nn)
        self.cs = (self.mueff + 2) / (self.nn + self.mueff + 5)
        self.c1 = 2 / ((self.nn + 1.3) ** 2 + self.mueff)
        self.cmu = min([1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((self.nn + 2) ** 2 + self.mueff)])
        self.damps = 1 + self.cs + 2 * max([0, ((self.mueff - 1)/self.nn) ** 0.5 - 1])  # ???

        self.pc, self.ps = np.zeros(self.nn), np.zeros(self.nn)
        self.B = np.eye(self.nn)
        self.D = np.ones(self.nn)
        self.C = np.eye(self.nn)
        self.M = np.eye(self.nn)

    def main_steps(self):
        # Sample
        self.D, self.B = np.linalg.eigh(self.C)
        self.D = self.D ** 0.5
        self.M = self.B * self.D
        newpop, z, d, fitvals = [], [], [], []
        for i in range(self.lam):
            zz = np.random.normal(0, 1, self.nn)
            dd = np.dot(self.M, zz)
            nn = self.xmean + self.sigma * dd
            z.append(zz), d.append(dd), newpop.append(nn)

        # sort and update mean
#        xmeanold = copy.deepcopy(self.xmean)
        fitvals = []
        for xx in newpop:
            fit = self.prob.evaluate(xx)
            self.ans.update(xx, fit)
            fitvals.append(fit)
        argx = np.argsort(fitvals)
        self.xmean = sum(self.weights[i] * newpop[argx[i]] for i in range(self.mu))  # Important

#        for i in range(self.nn):
#            f.write(str(self.xmean[i]) + ' ')
#        f.write('\n')

        # update evolution path
        zz = sum(self.weights[i] * z[argx[i]] for i in range(self.mu))
        c = (self.cs * (2 - self.cs) * self.mueff) ** 0.5
        self.ps -= self.cs * self.ps
        self.ps += c * zz
        dd = sum(self.weights[i] * d[argx[i]] for i in range(self.mu))
        c = (self.cc * (2 - self.cc) * self.mueff) ** 0.5
        self.pc -= self.cc * self.pc
        self.pc += c * dd

#        for i in range(self.nn):
#            f.write(str(dd[i]) + ' ')
#        f.write('\n')
#        for i in range(self.nn):
#            f.write(str(self.ps[i]) + ' ')
#        f.write('\n')
#        for i in range(self.nn):
#            f.write(str(self.pc[i]) + ' ')
#        f.write('\n')

        # update covariance matrix
        part1 = (1 - self.c1 - self.cmu) * self.C
        part2o = self.pc.reshape(self.nn, 1)
        part2t = self.pc.reshape(1, self.nn)
        part2 = self.c1 * np.dot(part2o, part2t)
        part3 = np.zeros((self.nn, self.nn))
        for i in range(self.mu):
            part3o = d[argx[i]].reshape(self.nn, 1)
            part3t = d[argx[i]].reshape(1, self.nn)
            part3 += self.cmu * self.weights[i] * np.dot(part3o, part3t)
        self.C = part1 + part2 + part3

#        c1a = self.c1
#        for i in range(self.nn):
#            for j in range(self.nn):
#                cmuij = sum(self.weights[k] * (newpop[argx[k]][i] - xmeanold[i]) \
#                            * (newpop[argx[k]][j] - xmeanold[j]) for k in range(self.mu)) / self.sigma ** 2
#                self.C[i][j] += (-c1a - self.cmu) * self.C[i][j] + self.c1 * self.pc[i] * self.pc[j] + self.cmu * cmuij

#        for i in range(self.nn):
#            for j in range(self.nn):
#                f.write(str(self.C[i][j]) + ' ')
#            f.write('\n')

        # update step-size
        self.sigma *= np.exp(min(0.6, (self.cs / self.damps) * (sum(x ** 2 for x in self.ps) / self.nn - 1) / 2))

    def optimize(self, maxgens):
        for i in range(maxgens):
            print("cmaesv2 ", i)
            self.main_steps()

