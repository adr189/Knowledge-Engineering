from probability import *
from prettytable import *
from utils import *
from utils import print_table
from solve_1_2_3_4 import *


################################################ INIT DIST FUNCTION ####################################################


def initDist(dimension, cells):
    initial = ProbDist('X0')
    if getCellsType(True, cells) and getDimensionType(True, dimension):
        for i in range(1, dimension + 1):
            initial[i] = getProb1D(True, cells, i) if i in cells else 0
    elif getCellsType(False, cells) and getDimensionType(True, dimension):
        for i in range(1, dimension + 1):
            initial[i] = getProb1D(False, cells, i) if i in cells else 0
    elif getCellsType(True, cells) and getDimensionType(False, dimension):
        for i in range(1, dimension[0] + 1):
            for j in range(1, dimension[1] + 1):
                initial[(i, j)] = getProb2D(True, cells, i, j) if (i, j) in cells else 0
    elif getCellsType(False, cells) and getDimensionType(False, dimension):
        for i in range(1, dimension[0] + 1):
            for j in range(1, dimension[1] + 1):
                initial[(i, j)] = getProb2D(False, cells, i, j) if (i, j) in cells else 0
    else:
        print("Something goes wrong")
    return initial


# Returns list if flag is true dict otherwise
def getCellsType(flag, cells):
    return isinstance(cells, list) if flag else isinstance(cells, dict)


# Returns int if flag is true tuple otherwise
def getDimensionType(flag, dimension):
    return isinstance(dimension, int) if flag else isinstance(dimension, tuple)


# Returns 1/len(cells) if flag is true cells[i] otherwise
def getProb1D(flag, cells, i):
    return 1 / len(cells) if flag else cells[i]


# Returns 1/len(cells) if flag is true cells[(i, j)] otherwise
def getProb2D(flag, cells, i, j):
    return 1 / len(cells) if flag else cells[(i, j)]


################################################### GO FUNCTION ########################################################


def printSorted(dicionario):
    print({k: v for (k, v) in sorted(dicionario.items())})


def go(celula, dim, movimentos, donut=False):
    moves_1d = {'E': 1, 'O': -1, '.': 0}
    moves_2d = {'E': (0, 1), 'O': (0, -1), '.': (0, 0), 'S': (1, 0), 'N': (-1, 0)}
    result = {}
    prob = 1 / len(movimentos)
    if getDimensionType(True, dim):
        for move in movimentos:
            m = moves_1d[move]
            sum = m + celula
            position = setDonut(1, dim, sum) if donut else setNotDonut(1, dim, sum)
            if position not in result:
                result[position] = 0
            result[position] += prob if getCellsType(True, movimentos) else movimentos[move]
    elif getDimensionType(False, dim):
        for move in movimentos:
            m = moves_2d[move]
            sum = celula[0] + m[0], celula[1] + m[1]
            position = (setDonut(1, dim[0], sum[0]), setDonut(1, dim[1], sum[1])) if donut else (
                setNotDonut(1, dim[0], sum[0]), setNotDonut(1, dim[1], sum[1]))
            if position not in result:
                result[position] = 0
            result[position] += prob if getCellsType(True, movimentos) else movimentos[move]
    else:
        print("Something goes wrong")
    return result


# Sets the donut rotation
def setDonut(min, max, position):
    if position == max + 1:
        return min
    elif position == min - 1:
        return max
    return position


# Holds the position (doesn't exist a donut)
def setNotDonut(min, max, position):
    if position == max + 1:
        return max
    elif position == min - 1:
        return min
    else:
        return position


################################################ FANTASMA FUNCTION #####################################################


def display(j):
    pretty = PrettyTable()
    aux = j.variables.copy()
    aux.append('Prob')
    pretty.field_names = aux
    for i in j.prob.keys():
        pretty.add_row(i + (j[i],))
    print(pretty)


def fantasmaConj(init, maxT, movimentos, donut=False):
    variables = getVariablesName(maxT)
    prob_dist = JointProbDist(variables)
    dim = list(init.prob)[-1]
    counter = 0
    list1 = []
    prob = []
    if getDimensionType(True, dim):
        fill(dim, counter, movimentos, list1, prob, init, maxT, prob_dist, donut)
    elif getDimensionType(False, dim):
        fill2D(dim, counter, movimentos, list1, prob, init, maxT, prob_dist, donut)
    else:
        print("Something goes wrong")
    return prob_dist


# Gets the variables names
def getVariablesName(max):
    var = []
    for i in range(0, max + 1):
        var.append("X" + str(i))
    return var


# Gets the moves using go function
def getMoves(counter, list, dim, moves, donut):
    if counter != 0:
        return go(list[len(list) - 1], dim, moves, donut)


# Appends probs to a list
def appendProb(counter, prob, init, move, i):
    if counter == 0:
        prob.append(init[i])
    else:
        prob.append(move[i] if i in move else 0)


# Delete Objects from the lists prob and list
def deleteObjects(prob, list):
    del prob[-1]
    del list[-1]


def fill2D(dim, counter, moves, list, prob, init, maxT, prob_dist, donut):
    move = getMoves(counter, list, dim, moves, donut)
    for i in range(1, dim[0] + 1):
        for j in range(1, dim[1] + 1):
            list.append((i, j))
            if counter == 0:
                prob.append(init[(i, j)])
            else:
                prob.append(move[(i, j)] if (i, j) in move else 0)

            if counter == maxT:
                k = 1
                for m in prob:
                    k *= m
                prob_dist[tuple(list)] = k
            else:
                fill2D(dim, counter + 1, moves, list, prob, init, maxT, prob_dist, donut)

            deleteObjects(prob, list)


# Fill the Joint Prob Dist
def fill(dim, counter, moves, list, prob, init, maxT, prob_dist, donut):
    move = getMoves(counter, list, dim, moves, donut)
    for i in range(1, dim + 1):
        list.append(i)

        appendProb(counter, prob, init, move, i)

        if counter == maxT:
            j = 1
            for k in prob:
                j *= k
            prob_dist[tuple(list)] = j
        else:
            fill(dim, counter + 1, moves, list, prob, init, maxT, prob_dist, donut)

        deleteObjects(prob, list)

################################################ PROBCOND FUNCTION #####################################################


def probCondFantasma(perg, cond, conjunta_completa):
    combined = dict(perg, **cond)
    vars_numer = [v for v in conjunta_completa.variables if not v in combined]
    vars_denom = [v for v in conjunta_completa.variables if not v in cond]
    return enumerate_joint(vars_numer, combined, conjunta_completa) / enumerate_joint(vars_denom, cond, conjunta_completa)

################################################ MAXPROB FUNCTION #####################################################

def maxProb(conjunta, i):
    probabkeys = list(conjunta.prob) #lista de keys
    probabvalues = list(conjunta.prob.values()) #lista de values
    probabvalueslength = len(probabvalues)
    tuple1 = []
    tuple2 = []
    tuple3 = []
    tuple4 = []
    tuple5 = []
    outputtuple = []
    
    for k in range(probabvalueslength):
        if probabkeys[k][i] == 1:
            tuple1.append(k)
            count1 = 0
            for l in range(len(tuple1)):
                count1+= probabvalues[tuple1[l]]
        elif probabkeys[k][i] == 2:
            tuple2.append(k)
            count2 = 0
            for l in range(len(tuple2)):
                count2+= probabvalues[tuple2[l]]
        elif probabkeys[k][i] == 3:
            tuple3.append(k)
            count3 = 0
            for l in range(len(tuple3)):
                count3+= probabvalues[tuple3[l]]
        elif probabkeys[k][i] == 4:
            tuple4.append(k)
            count4 = 0
            for l in range(len(tuple4)):
                count4+= probabvalues[tuple4[l]]
        elif probabkeys[k][i] == 5:
            tuple5.append(k)
            count5 = 0
            for l in range(len(tuple5)):
                count5+= probabvalues[tuple5[l]]
        
    maxprobability = max(count1, count2, count3, count4, count5)
    if maxprobability == count1:
        outputtuple.append(1);
    elif maxprobability == count2:
        outputtuple.append(2);
    elif maxprobability == count3:
        outputtuple.append(3);
    elif maxprobability == count4:
        outputtuple.append(4);
    elif maxprobability == count5:
        outputtuple.append(5);
    return outputtuple, maxprobability

################################################ MAXPROB TESTS #####################################################

ini_1D = initDist(5, {1:0.6,3:0.4})
f_1D = fantasmaConj(ini_1D, 6, {'E':0.3, 'O':0.2,'.':0.5})
t = maxProb(f_1D, 3)
print("{}, {:.6f}".format(t[0],t[1]))

ini_1D = initDist(5, {1:0.6,3:0.4})
f_1D = fantasmaConj(ini_1D, 6, {'E':0.3, 'O':0.2,'.':0.5})
print(maxProb(f_1D, 3))

ini_1D = initDist(5, [1,5])
f_1D = fantasmaConj(ini_1D, 6, ['O','E', '.'])
print(maxProb(f_1D, 4))

#####################################################################################################################

ini_1D = initDist(5, {1:0.6,3:0.4})
f_1D = fantasmaConj(ini_1D, 6, {'E':0.3, 'O':0.2,'.':0.5})
query = dict(X0=3,X1=3,X2=3,X3=3)
evidence = dict()
print("{:.6f}".format(probCondFantasma(query, evidence, f_1D)))
