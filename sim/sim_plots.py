import numpy as np
import matplotlib.pyplot as plt

def plotTermStructure(termStructurePath, maturities, dateIndices = [0]):
    for i in dateIndices:
        plt.plot(maturities, termStructurePath[i], label=f'Path {i}')
    plt.xlabel('Maturity')
    plt.ylabel('Term Structure')
    plt.legend()
    plt.show()

def plotYield(termStructurePath, maturities, tenors):
    for tenor in tenors:
        plt.plot(termStructurePath[:, np.where(maturities == tenor)[0]], label=f'Tenor {tenor}')
    plt.xlabel('time')
    plt.ylabel('yield')
    plt.title('Yield for multiple tenors')
    plt.legend()
    plt.show()