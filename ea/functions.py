class Sphere:

    dim = 50
    lb, ub = -100, 100

    @staticmethod
    def evaluate(xx):
        re = sum(xx[i] ** 2 for i in range(Sphere.dim))
        return re


class Elli:

    dim = 5
    lb, ub = 0.5, 0.5

    @staticmethod
    def evaluate(xx):
        n = len(xx)
        aratio = 1e3
        re = sum(xx[i] ** 2 * aratio ** (2. * i / (n - 1)) for i in range(n))
        return re


class Rosenbrock:

    dim = 50
    lb, ub = 1, 1

    @staticmethod
    def evaluate(xx):
        re = 0
        for i in range(Rosenbrock.dim - 1):
            re += 100 * (xx[i] ** 2 - xx[i + 1]) ** 2 + (xx[i] - 1) ** 2
        return re

