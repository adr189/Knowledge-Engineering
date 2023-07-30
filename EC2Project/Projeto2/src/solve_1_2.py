################################################### IMPORTS ############################################################


from builtins import dict
from funcoesFantasma import *
from probabilityPlus import *


################################################# FANTASMA_RB ##########################################################


# Gets the list of all variables
def get_variables(max):
    var = []
    for i in range(max + 1):
        var.append('X' + str(i))
    return var


# Gets the list of ghost movement probs
def getProbs(distIni, dim, movimentos, donut):
    prob = {}
    for i in distIni.values:
        prob[str(i)] = go(i, dim, movimentos, donut)
    return prob


# Gets the list of all cells
def getCells(distIni):
    cells = []
    for i in distIni.values:
        cells.append(i)
    return cells


# Creates an empty list of list with the cells size
def createList(cells):
    return [[None for i in range(len(cells))] for j in range(len(cells))]


# Fills the movement probs
def fill_probs(probs, cells):
    listt = createList(cells)
    for i in range(len(listt)):
        od = collections.OrderedDict(sorted(probs.get(list(probs)[i]).items()))
        keys = list(od.keys())
        values = list(od.values())
        k = 0
        for j in range(len(listt)):
            if cells[j] in keys:
                listt[i][j] = values[k]
                k += 1
            else:
                listt[i][j] = 0
    return listt


# Gets the dictionary zipping the cells and the probs
def getDict(cells, listt):
    l1 = []
    for i in cells:
        l1.append(str(i))
    res = []
    for i in range(len(l1)):
        res.append(dict(zip(l1, listt[i])))
    d1 = {}
    for i in range(len(l1)):
        d1[l1[i]] = res[i]
    return d1


# Puts the int position in string position
def distIniToString(distIni):
    p = {}
    for i in distIni.prob:
        p[str(i)] = distIni.prob.get(i)
    return p


# Fills the Bayes Net
def fill_rb(distIni, max, movimentos, donut, dim):
    probs = getProbs(distIni, dim, movimentos, donut)
    cells = getCells(distIni)
    listt = fill_probs(probs, cells)
    d1 = getDict(cells, listt)
    rb = RedeBayes()
    rb.add(('X0', '', distIniToString(distIni)))
    for i in range(1, max + 1):
        rb.add((get_variables(max)[i], get_variables(max)[i - 1], d1))
    return rb


# Returns an instance of Bayes Net.
def fantasma_RB(distIni, tmax, movimentos, donut=False):
    dim = list(distIni.prob)[-1]
    return fill_rb(distIni, tmax, movimentos, donut, dim)


############################################### FANTASMA_REPEAT ########################################################


# Fills the only 2 nodes Bayes Net
def fill_2n_rb(distIni, max, movimentos, donut, dim):
    var = get_variables(max)
    probs = getProbs(distIni, dim, movimentos, donut)
    cells = getCells(distIni)
    listt = fill_probs(probs, cells)
    d1 = getDict(cells, listt)
    rb = RedeBayes([('Xt', '', distIniToString(distIni)), ('Xt+1', 'Xt', d1)])
    i = 0
    while i < max:
        rb.variable_node('Xt').cpt[()] = enumeration_ask('Xt+1', {}, rb)
        i += 1
    return ProbDist(var[max], enumeration_ask('Xt', {}, rb).prob)


# Returns an instance of Dist Prob
def fantasma_repeat(distIni, max, movimentos, donut=False):
    dim = list(distIni.prob)[-1]
    return fill_2n_rb(distIni, max, movimentos, donut, dim)

