import functions
import cmaes
import cmaesv2
import maes
import matplotlib.pyplot as plt


def printf(name, opti):
    file = open(name, "w")
    for i in range(opti.ans.evals):
        file.write(str(i) + ' ' + str("%.6f" % opti.ans.fit[i]) + '\n')
    file.close()


if __name__ == "__main__":
    #opt1 = cmaes.CMAES(functions.Rosenbrock)
    #opt1.optimize(600)
    # printf("Result_cmaes.txt", opt1)
    # opt2 = cmaesv2.CMAES(functions.Rosenbrock)
    # opt2.optimize(600)
    # printf("Result_cmaesv2.txt", opt2)
    opt3 = maes.MAES(functions.Rosenbrock)
    opt3.optimize(600)
    printf("Result_maes.txt", opt3)
    # plt.figure("compare")
    # xx = range(opt1.ans.evals)
    # plt.plot(xx, opt1.ans.fit, color='black', linewidth=1.0)
    # plt.plot(xx, opt2.ans.fit, color='blue', linewidth=1.0)
    # plt.plot(xx, opt3.ans.fit, color='red', linewidth=1.0)
    # plt.show()


