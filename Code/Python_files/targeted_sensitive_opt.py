""" Solves the optimization problem of targeted Sensitive Pagerank.

It solves the targeted optimization problem as described
in "Fairness-Aware Link Analysis"[1] paper. It always chooses the top
k node based to pagerank ranking. Makes use of cvx
optimization package[2].

Parameters:
    phi (float): Wanted ratio for the protected category
        (category 1 - R). If phi == 0 => phi = ratio of protected
        category (i.e. |R|/|N|). N:= Set of alla nodes.
        R:= set of red nodes.
    k (int): Number of nodes to take into consideration.

Creates:
    "out_sensitive_jump_v.txt" (txt file): Μinimizer vector.
    "out_sensitive_pagerank.txt" (txt file): Sensitive Pagerank
        corresponds to minimizer vector.

References:
    [1]
    [2] https://cvxopt.org/

TODO:
    Add reference to paper.
"""
import numpy as np
import cvxopt as cvx
from cvxopt import spmatrix, matrix, solvers
from scipy.sparse import coo_matrix
import sys

def get_red_ratio(index):
    """ Returns the ration of protected category."""

    # index[i] == true if node i belongs to protected category. 
    ratio = sum(index) / len(index)

    return ratio

def uniformPR(M, gamma = 0.15):
    """Returns the Pagerank and the Q matrix.
    
    Q: P(v) = Qv and v is the unifrom vector (i.e. v(i) -= 1/N).
    N:= number of nodes.

    Parameters:
        M (nd.array): Adjacency Matrix.
        gama (float): Probability to jump. Jump vector's coafficient.

    Returns:
        p (1D np.array): Pagerank vector.
        Q (2D nd.array): P(v) = Qv.
    """
    n = M.shape[0]
    d = np.array(np.reciprocal(M.sum(axis = 1).T))
    d = d.flatten()
    D = np.diag(np.array(d))
    P = D.dot(M)
    Q = gamma*np.linalg.inv((np.eye(n) - (1-gamma)*P))
    Q = Q.T
    u = (np.ones(n)/(1.0*n)).T
    p = Q.dot(u)

    return (p,Q)

def new_index(p, index):
    n = len(p)
    sorted_index = np.argsort(-p)

    #keep only the rows of top-K nodes
    top_k = [False for i in range(n)]
    for i in sorted_index[:k]:
        top_k[i] = True # True if node is in topk results

    index_new = [top_k[i] and index[i] for i in range(n)]

    return index_new, top_k

def fairPR(M, index, phi, k):
    p,Q = uniformPR(M)
    index_top_k_1, index_top_k = new_index(p, index)#true if in protected category(1) and on top-k.
    n = p.size
    G = matrix([matrix(-1*np.eye(n))])
    h = matrix([matrix(np.zeros(n))]) # inequality constraint seems correct
    A = matrix([matrix((Q[index_top_k_1,:].sum(0) - (phi * Q[index_top_k,:]).sum(0))),matrix(np.ones(n))],(n,2)).T
    b = matrix([0.,1.])
    Q = matrix(Q)
    p = matrix(p)
    x = solvers.qp(P = Q.T*Q, q=-Q.T*p, G=G, h=h,A=A,b=b)['x'] # seems correct

    p, Z = uniformPR(M)
    check_solution(index, p, Q*x, k)
    
    return (x,Q*x) # optimal jump vector, corresponding pagerank
   
def create_adj_matrix(filename = "out_graph.txt"):
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

if len(sys.argv) != 3:
    print("provide 2 arguments <ratio for protected category>, <top-k>")
else:
    pr = float(sys.argv[1])
    k = int(sys.argv[2])


M, index = create_adj_matrix()

if pr == 0:
    pr = get_red_ratio(index)

j, fp = fairPR(M,index, pr, k)

j = np.array(j).flatten()
fp = np.array(fp).flatten()

with open("out_sensitive_topk_jump_v.txt", "w") as file_one:
    for i in j:
        file_one.write(str(i) + "\n")
with open("out_sensitive_topk_pagerank.txt", "w") as file_one:
    for i in fp:
        file_one.write(str(i) + "\n")



