""" Find the optimal solution of targeted algorithms.

Creates:
    "out_sensitive_jump_v.txt" (txt file): Μinimizer vector.
"""
import numpy as np
import cvxopt as cvx
from cvxopt import spmatrix, matrix, solvers
from scipy.sparse import coo_matrix
import sys

def uniformPR(n):
    """Returns the Pagerank vector

    It reads the pagerank vector from the txt file generated by
    pagerank.out
    """
    vec = np.zeros(n)
    with open("out_pagerank_pagerank.txt", "r") as file_one:
        for i in range(n):
            vec[i] = float(file_one.readline())
    return vec

def new_index(p, index):
    """ Return top-k indexes.

    Returns:
        index_new (pytho list): index_new[i] == true if node i is red
            and is in top-k nodes of pagerank.
        top_k (python list): top_k[i] == true if node i is in top-k
            nodes of pagerank.
    """
    n = len(p)
    sorted_index = np.argsort(-p)

    #keep only the rows of top-K nodes
    top_k = [False for i in range(n)]
    for i in sorted_index[:k]:
        top_k[i] = True # True if node is in topk results

    index_new = [top_k[i] and index[i] for i in range(n)]

    return index_new, top_k

def create_adj_matrix(filename = "out_graph.txt"):
    """ Creates Adjacency matrix and index list.

    Parameters:
        edge_file (txt file): edge list file in proper format.
        com_file (txt file): community file in proper format.

    Returns:
        M (nd.array): Adjacency Matrix.
        index (python list): index[i] == True if node i belongs to protected
            category (i.e. category 1).
    
    Notes:
        See specifications for the files in general description
        of the project.
        
    TODO: Add link for the general specifications.
    """
    n = 0
    with open(filename, "r") as file_one:
        n = int(file_one.readline())

    M = np.zeros((n,n))

    with open(filename, "r") as file_one:
        file_one.readline()
        for line in file_one:
            edge = line.split()
            M[int(edge[0])][int(edge[1])] = 1.
            
    index = [False for i in range(n)]
    with open("out_community.txt", "r") as file_one:
        file_one.readline()
        for line in file_one:
            info = line.split()
            if int(info[1]) == 1:
                index[int(info[0])] = True

    j = 0
    for i in range(n):
        if not M[i].any():
            j += 1
            M[i] = [1. for k in M[i]]
    print("%d vectors without out neighbors" %j)

    return M, index

def fairPR(M, index, phi, k):
    """ Returns the optimal Pagerank for targeted algorithm.

    Parameters:
        M (nd.array): Adjacency Matrix.
        index (python list): index[i] == True if node i belongs
            to protected category (i.e. category 1). 
        phi (float): Wanted ratio for protected category.
        k (int): Number of nodes for the targeted algorithm.

    Returns:
        p (1D np.array): Optimal pagerank vector.
    """
    p = uniformPR(len(index))
    index_top_k_1, index_top_k = new_index(p, index)
    n = p.size
    Q = np.eye(n)
    G = matrix([matrix(-1*np.eye(n))])
    h = matrix([matrix(np.zeros(n))])
    A = matrix([matrix((Q[index_top_k_1,:].sum(0) - (phi * Q[index_top_k,:]).sum(0))),matrix(np.ones(n))],(n,2)).T
    b = matrix([0.,1.])
    Q = matrix(Q)
    p = matrix(p)
    x = solvers.qp(P = matrix(np.eye(n)), q=-Q.T*p, G=G, h=h,A=A,b=b)['x']

    x = np.array(x).flatten()

    return x
 
# Read Command line arguments.
if len(sys.argv) != 3:
    print("provide 2 arguments <ratio for protected category>, <top-k>")
else:
    pr = float(sys.argv[1])
    k = int(sys.argv[2])

# Get Adjacency Matrix and index vector
# (index[i] == true if node i belongs to protected category).
M, index = create_adj_matrix()

if pr == 0:
    pr = sum(index) / len(index)
print("phi: ", pr)

# Get optimal targeted vector.
top_k_p = fairPR(M,index, pr, k)
top_k_p = np.array(top_k_p).flatten()# Probeblywe don't need this.

# Store results in text files.
with open("out_targeted_optimal.txt", "w") as file_one:
    for i in top_k_p:
        file_one.write(str(i) + "\n")