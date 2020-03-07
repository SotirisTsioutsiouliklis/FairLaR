""" Solves a variation of optimization problem of targeted Pagerank.

It solves the targeted optimization problem as described
in "Fairness-Aware Link Analysis"[1] paper but for the topk of the
pagerank algorithm. It also preserve the order of these nodes compare
to simpletargeted. Makes use of cvx optimization package[2].

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

def uniformPR(M, gamma = 0.15):
    """ Returns the Pagerank and the Q matrix.
    
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
    P = D.dot(M) # up to here xorrect - M must have float elements
    Q = gamma*np.linalg.inv((np.eye(n) - (1-gamma)*P))
    Q = Q.T# up to here also correct
    u = (np.ones(n)/(1.0*n)).T
    p = Q.dot(u) # also correct
    with open("opt_pagerank.txt", "w") as file_one:
        for value in p:
            file_one.write("%f\n" %value)
    return (p,Q) # Returns pagerank vector and Q matrix

def top_k(p, Q, k):
    """ Creates matrix used for inequality constraints.

    Matrix K describes the constraint that is responsible to keep the
    k nodes in the top k positions. 

    Returns:
        K (2D nd.array): Constraint matrix.
    """
    n = len(p)
    sorted_index = np.argsort(-p)

    # calculate matrix K
    K = np.zeros((k*(n-k), n))
    k_row = 0
    for i in sorted_index[:k]:
        for j in sorted_index[k:]:
            K[k_row,:] = Q[int(j),:] - Q[int(i),:]
            k_row += 1

    return K    

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

def fairPR(M, index, phi, k):
    """ Returns minimizer jump vector, corresponding pagerank.

    Parameters:
        M (nd.array): Adjacency Matrix.
        index (python list): index[i] == True if node i belongs
            to protected category (i.e. category 1). 
        phi (float): Wanted ratio for protected category.
        k (int): Number of nodes for the targeted algorithm.

    Returns:
        x (CVX 1D matrix): Minimizer jump vector.
        Q*x (CVX 1D matrix): Optimal pagerank vector.
    """
    p,Q = uniformPR(M)
    K = top_k(p, Q, k)
    index_top_k_1, index_top_k = new_index(p, index)
    n = p.size
    G = matrix([matrix(-1*np.eye(n)), matrix(K)])
    h = matrix([matrix(np.zeros(n)), matrix(np.zeros(k*(n-k)))])
    A = matrix([matrix((Q[index_top_k_1,:].sum(0) - (phi * Q[index_top_k,:]).sum(0))),matrix(np.ones(n))],(n,2)).T
    b = matrix([0.,1.])
    Q = matrix(Q)
    p = matrix(p)
    x = solvers.qp(P = Q.T*Q, q=-Q.T*p, G=G, h=h,A=A,b=b)['x']

    p, Z = uniformPR(M)
    
    return (x,Q*x)
   
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

# Read command line arguments.
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

# Get fair pagerank and corresponding jump vector.
j, fp = fairPR(M,index, pr, k)

# Change from data type cvx matrix to numpy array.
j = np.array(j).flatten()
fp = np.array(fp).flatten()

# Store results in text files.
with open("out_targeted_topk_jump_v.txt", "w") as file_one:
    for i in j:
        file_one.write(str(i) + "\n")
with open("out_stargeted_topk.txt", "w") as file_one:
    for i in fp:
        file_one.write(str(i) + "\n")