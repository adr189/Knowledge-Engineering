################################################### IMPORTS ############################################################
from builtins import dict

from typing import List
from funcoesFantasma import *
from probabilityPlus import *
from prettytable import *
from utils import *
from solve_1_2 import *


################################################ FANTASMA_TRACK ########################################################


def fantasma_track(initDist, landmarks, sensor, periodo, leituras, movimentos, donut=False, verbose=False):
    return 0


#################################################### TESTS #############################################################


# TEST1
ini = initDist((3,3),[(1,1)])
f=fantasma_repeat(ini,5,['E','.'])
printSorted(f.prob)
#
# TEST2
ini = initDist(20,[1])
f=fantasma_repeat(ini,1000,['O','E','.'],True)
printSortedRound(f.prob,5)
#
# TEST3
ini = initDist((10,10),[(1,1)])
f=fantasma_repeat(ini,50,['E','S','.'])
v=list(f.prob.values())
print('{0:.10f}'.format(max(v)))
#
# TEST4
ini = initDist(50,[1])
f=fantasma_repeat(ini,60,{'E':0.9,'.':0.1},False)
v=list(f.prob.values())
print('{0:.10f}'.format(max(v)))
