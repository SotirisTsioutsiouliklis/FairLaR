""" Compares the optimization algorithms.

This script has purpose to compare my stochastic optimization algorithm
with the cvxopt one. And my stochastic residual algorithm with the two
gradient based.

Objectives:
    1. Valide results.
        a. Probability interpretation
        b. Fair rank vector.
    2. Distance from pagerank vector.
    3. Running time (Pending...)
"""
import numpy as np
import sys
from scipy.spatial import distance

class RankVector:
    """ This class represent a rank vactor.

    Public attributes
    -----------------
    None

    Private attributes
    ------------------
    _numberOfNodes (int): Pending...
    _rankVector (np.array): Pending...

    Public Methods
    --------------
    getNumberOfNodes (int): Number of Nodes. 
    getRankVector (np.array): Rank vector.

    Private Methods
    ---------------
    Pending...

    """
    # Constractor.
    def __init__(self, name):
        self._numberOfNodes = self.__readNumberOfNodes()
        self._rankVector = self.__readVector(name)

    # Getters-Setters.
    #
    def getNumberOfNodes(self):
        return self._numberOfNodes
    
    #
    def getRankVector(self):
        """ Returns the rank vector.

        Returns rankVector(np.array)
        """ 
        return self._rankVector

    # Private Methods.
    #
    def __readNumberOfNodes(self):
        n = 0
        with open("out_graph.txt", "r") as file_one:
            n = int(file_one.readline())
            
        return n

    #
    def __readVector(self, name):
        """ Reads rank vector from txt file.

        Returns:
            rankVector(np.array): The rank vector for the specified
                algorithm.
        """
        rankVector = np.zeros(self._numberOfNodes)
        with open(name, "r") as file_one:
            node = 0
            for line in file_one:
                value = float(line)
                rankVector[node] = value
                node += 1
        
        return rankVector
    
    #
    def isValide(self):
        valide = True
        sum = 0
        for i in self._rankVector:
            if i < 0:
                valide = False
            sum += i
        if (sum > 1 + 10 **(-4)):
            valide = False

        return valide

class Algorithms:
    """ A collection of the results of various algorithms. And methods
    for basic evaluation - comparison.
    """
    def __init__(self):
        self._algorithms = dict()
        self._numberOfAlgorithms = 0

    # Getters-Setters.
    #
    def getAlgorithms(self):
        return self._algorithms

    #
    def getAlgorithm(self, algoName):
        return self._algorithms[algoName]

    # Public methods.
    #
    def append(self, algoName, fileName):
        self._algorithms[algoName] = RankVector(fileName)
        self._numberOfAlgorithms += 1

    #
    def remove(self, algoName):
        self._algorithms.pop(algoName)
        self._numberOfAlgorithms -= 1

    #

class Evaluate():
    """ Basic methods to evaluate different fair rank algorithms.
    """
    def __init__(self, algorithms):
        self._algorithms = algorithms

    def getAlgorithms(self):
        return self._algorithms

    def getObjectiveValues(self):
        """ L2 Distance from PageRank vector.

        Returns
        -------
        objective_values.txt: Pending...
        """
        pagerank = self._algorithms.getAlgorithm("pagerank").getRankVector()

        values = dict()

        # Calculate Values.
        for algorithm, rankVector in self._algorithms.getAlgorithms().items():
            values[algorithm] = distance.euclidean(pagerank, rankVector.getRankVector())
            #values[algorithm] = np.linalg.norm(pagerank - rankVector.getRankVector())

        # Write Results.
        with open("objective_results.txt", "w") as file_one:
            file_one.write("Algorithm\tValue\n")
            for algorithm, value in values.items():
                file_one.write(algorithm + "\t" + str(value) + "\n")

    def checkValideRankVectors(self):
        for name, rankVector in self._algorithms.getAlgorithms().items():
            if not rankVector.isValide():
                print(name + " Nor valide rank vector")
    
#--------------------------- Main ----------------------

items = [("pagerank", "out_pagerank_pagerank.txt"),
             ("sensitive", "out_sensitive.txt"),
             ("stochastikSensitive", "opt_mine_fairpagerank_vector.txt"),
             ("stochastikExcess", "out_excess_sensitive_pagerank.txt"),
             ("gradientExcess", "out_grad_excess_sensitive_pagerank.txt"),
             ("gradientExessProj", "out_proj_excess_sensitive_pagerank.txt"),
             ("localNeighborhood", "out_lfpr_n_pagerank.txt"),
             ("localProportional", "out_lfpr_p_pagerank.txt"),
             ("localUniform", "out_lfpr_u_pagerank.txt")]

algorithms = Algorithms()
for i in items:
    try:
        algorithms.append(i[0], i[1])
    except:
        print(i[0] + " didn't load properly\n")

analyst = Evaluate(algorithms)
analyst.getObjectiveValues()
analyst.checkValideRankVectors()