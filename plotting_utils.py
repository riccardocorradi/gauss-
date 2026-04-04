import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as mtick

def residuals_actualFitted(fittingErrors, fittedTs, termStructurePath, tenor,
                           figsize = (20,5)):
    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (20, 5))
    ax[0].plot(fittingErrors[tenor], color = 'blue')
    ax[1].plot(fittedTs[tenor], label = 'fitted', color = 'blue')
    ax[1].plot(termStructurePath[tenor], label = 'actual', color = 'red')

    ax[0].yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f'{y*100:.0f}'))
    ax[0].set_ylabel('bps')
    ax[1].yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f'{y/100:.2%}'))
    ax[1].legend()
    ax[0].set_title('fitting error')
    ax[1].set_title('actual versus fitted yields')

def multipleResiduals(fittingErrors, tenors, figsize = (20,5)):
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (20, 5))
    for t in tenors:
        ax.plot(fittingErrors[t], label = t)

    ax.legend()
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f'{y*100:.0f}'))
    ax[0].set_ylabel('bps')