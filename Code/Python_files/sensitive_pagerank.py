""" Solves the optimization problem of Sensitive Pagerank.

If P(v) = Qv. It finds the P'(v') that minimize the function
||P(v) - P'(v')|| where P' is fair in the notion described
in "Fairness-Aware Link Analysis"[1] paper. Makes use of cvx
optimization package[2].

Parameters:
    phi (float): Wanted ratio for the protected category
        (category 1 - R). If phi == 0 => phi = ratio of protected
        category (i.e. |R|/|N|). N:= Set of alla nodes.
        R:= set of red nodes.

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

def fairPR(M,index,phi):
    """ Returns optimal jump vector and corresponding pagerank.

    Parameteres:
        M (nd.array): Adjacency Matrix.
        index (python list): index[i] == True if node i belongs to protected
            category (i.e. category 1). 

    Returns:
        x (1D cvx matrix): Minimizer of the optimization problem.
        P_fair (1D cvx matrix): Pagerank corresponds to minimizer
            of the optimization problem.
    """
    # Get pagerank and W matrix.
    p,Q = uniformPR(M)
    #test for solution.
    test_for_solution(Q, index, phi)
    # Get size.
    n = p.size
    # Inequality constraint.
    G = matrix([matrix(-1*np.eye(n))])
    h = matrix([matrix(np.zeros(n))])
    # Equality constraints.
    A = matrix([matrix(Q[index,:].sum(0)),matrix(np.ones(n))],(n,2)).T
    b = matrix([phi,1])
    # Change data type to cvx matrix.
    Q = matrix(Q)
    p = matrix(p)
    # Call solver.
    x = solvers.qp(P = Q.T*Q, q=-Q.T*p, G=G, h=h,A=A,b=b)['x']
    
    return (x,Q*x)

def test_for_solution(Q, index, phi):
    """ Checks if there is feasible solution.

    Methodology is described at "Fairness-Aware Link Analysis" papaer.
    Terminates the program if there is no feasible solution. Otherwise
    it does nothing.

    Parameters:
        Q (2D nd.array): P(v) = Qv.
        index (python list): index[i] == true if node i belongs
            to protected category.
        phi (float): Wanted ratio for the protected category.

    exits:
        If no possible solution.
    """
    n = len(index)
    q = np.zeros(n)
    for i in range(n):
        if index[i]:
            q[i] = 1.
    
    h = q.dot(Q)
    exist_solution_0 = False
    exist_solution_1 = False
    for i in h:
        if i >= phi:
            exist_solution_0 = True
            break
    for i in h:
        if i <= phi:
            exist_solution_1 = True
            break
    if not (exist_solution_0 and exist_solution_1):
        sys.exit("no solution exists")

    return q

def create_adj_matrix(edge_file = "out_graph.txt", com_file="out_community.txt"):
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
    with open(edge_file, "r") as file_one:
        n = int(file_one.readline())

    M = np.zeros([n,n])

    with open(edge_file, "r") as file_one:
        file_one.readline()
        for line in file_one:
            edge = line.split()
            M[int(edge[0])][int(edge[1])] = 1.
            
    index = [False for i in range(n)]
    with open(com_file, "r") as file_one:
        file_one.readline()
        for line in file_one:
            info = line.split()
            if int(info[1]) == 1:
                index[int(info[0])] = True

    j = 0
    for i in range(n):
        if not M[i].any():
            j += 1
            M[i] = [1. for i in M[i]]

    return M, index

# Read the command line argument.
if len(sys.argv) != 2:
    print("provide 1 arguments <ratio for protected category>")
else:
    pr = float(sys.argv[1])

# Get Adjacency Matrix and index vector
# (index[i] == true if node i belongs to protected category).
M,index = create_adj_matrix()

if pr == 0:
    pr = get_red_ratio(index)
print("phi: ", pr)

# Get fair pagerank and corresponding jump vector.
j,fp = fairPR(M,index, pr)

# Change from data type cvx matrix to numpy array.
j = np.array(j).flatten()
fp = np.array(fp).flatten()

# Store results in text files.
with open("out_sensitive_jump_v.txt", "w") as file_one:
    for i in j:
        file_one.write(str(i) + "\n")
with open("out_sensitive_pagerank.txt", "w") as file_one:
    for i in fp:
        file_one.write(str(i) + "\n")
