from util.plots import *
import numpy as np, math
from functools import partial

"""Grow back % after drop"""
def growBack(drop):
    return 100 * drop / (100 - drop)
def timeToGrowBack(drop, growRate):
    return np.log(1+growBack(drop)/100, 1+growRate/100)

# utilPlotFunc(growBack, 0 , 100, 1, label='How much % grow back after drop (x)')
# growRate = 2
# utilPlotFunc(partial(timeToGrowBack, growRate=growRate), 0, 100, .1, label='Years of growth with rate {} if drop=x'.format(growRate))
utilPlot3Dfunc(timeToGrowBack, 1,100, 1,7,1,.1)
