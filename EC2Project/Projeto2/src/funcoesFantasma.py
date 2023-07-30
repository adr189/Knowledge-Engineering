from probabilityPlus import *

# Exemplos
# >>ini = initDist(5,[1,5])
# >>print(ini.prob)
# {1: 0.5, 2: 0, 3: 0, 4: 0, 5: 0.5}

def initDist(dimensao, celulas):
    """ Define a distribuição inicial com base em parâmetros de input.
        Gera mesmo uma instância da classe ProbDist.
        A dimensão pode ser apenas um inteiro (1..input)
        Pode ser 2D e nesse caso é um tuplo (NLinhas, NColunas) começa em (1,1)
        Se células for uma lista a distribuição é uniforme para os elementos dados
        Se for um dicionário, esses ficarão com as probabilidades dadas
        Todos os outras casas do espaço ficarão com probabilidade apriori de 0.
    """
    freq = {}
    if isinstance(dimensao,tuple) :
        if isinstance(celulas,list) :
            prob = 1/len(celulas)
            for i in range(1,1+dimensao[0]):
                for j in range(1,1+dimensao[1]):
                    if (i,j) in celulas :
                        freq[(i,j)] = prob
                    else :
                        freq[(i,j)] = 0
        else :
            for i in range(1,1+dimensao[0]):
                for j in range(1,1+dimensao[1]):
                    if (i,j) in celulas :
                        freq[(i,j)] = celulas[(i,j)]
                    else :
                        freq[(i,j)] = 0
    else:
        if isinstance(celulas, list) :
            prob = 1/len(celulas)
            for i in range(1,1+dimensao) :
                if i in celulas :
                    freq[i] = prob
                else :
                    freq[i] = 0
        else :
            for i in range(1,1+dimensao) :
                if i in celulas :
                    freq[i] = celulas[i]
                else :
                    freq[i] = 0    
    return ProbDist('X0',freq)




def go(celula, dim, movimentos, donut = False):
    """ Dada uma célula num espaço limitado por dim (limites de 1D ou 2D)
        e uma representação das acções (N,S,E,O,.) e indicação de "dar a volta"
        em donut, devolve a distribuição de probabilidade para as casas no instante seguinte
        que não tenham probabilidade 0.
    """
    
    dicSeguintes = {}
    for mov in movimentos:
        celSeguinte, probSeguinte = moveSeguinte(celula, dim, mov, movimentos, donut)
        if celSeguinte in dicSeguintes:
            dicSeguintes[celSeguinte] = dicSeguintes[celSeguinte] + probSeguinte
        else:
            dicSeguintes[celSeguinte] = probSeguinte
    return dicSeguintes

def moveSeguinte(celula, dim, mov, movimentos, donut):
    if type(movimentos) == list:
        prob = 1/ len(movimentos)
    else:
        prob = movimentos[mov]
    if mov == '.':
        return (celula, prob)
    elif type(dim) == int:
        return (celulaSeguinte1D(celula, dim, mov, donut), prob)
    else:
        return (celulaSeguinte2D(celula, dim, mov, donut), prob)
    
def celulaSeguinte1D(celula, dim, mov, donut):
    if mov == 'E':
        if celula == dim:
            if not donut:
                return celula
            else:
                return 1
        else:
            return celula + 1
    else: # mov == 'O'
        if celula == 1:
            if not donut:
                return celula
            else:
                return dim
        else:
            return celula - 1
        
   
def celulaSeguinte2D(celula, dim, mov, donut):
    linhaCorrente, colunaCorrente = celula
    dimLinhas , dimColunas = dim
    if mov == 'E' or mov == 'O':
        return  (linhaCorrente, celulaSeguinte1D(colunaCorrente, dimColunas, mov, donut) )
    elif mov == 'N':
        if linhaCorrente == 1:
            if not donut:
                return celula
            else:
                return [dimLinhas, colunaCorrente]
        else:
            return (linhaCorrente - 1, colunaCorrente)
    else: #mov == 'S'
        if linhaCorrente == dimLinhas:
            if not donut:
                return celula
            else:
                return (1, colunaCorrente)
        else:
            return (linhaCorrente + 1, colunaCorrente)


# Imprime um dicionário, ordenado pelas chaves      
def printSorted(dicionario):
    print({k:v for (k,v) in sorted(dicionario.items())})

# Imprime um dicionário, ordenado pelas chaves, mas com os valores numéricos arredondados 
def printSortedRound(dicionario,casas):
    """ Requires: dicionário com valores numéricos e chaves que se possam ordenar e as casas é um inteiro
            Ensures: uma apresentação do dicionário ordemado pelas chaves e com os valores arrendondados 
            ao número de casas decimais.
    """
    print({k:round(v,casas) for (k,v) in sorted(dicionario.items())})