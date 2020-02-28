# TODO: Add documentation.
import numpy as np
import cvxopt as cvx
from cvxopt import spmatrix, matrix, solvers
from scipy.sparse import coo_matrix
import sys

def uniformPR(M, gamma = 0.15):
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
    n = len(p)
    sorted_index = np.argsort(-p)

    #keep only the rows of top-K nodes
    top_k = [False for i in range(n)]
    for i in sorted_index[:k]:
        top_k[i] = True # True if node is in topk results

    index_new = [top_k[i] and index[i] for i in range(n)]

    return index_new, top_k

#solves min||Qv - P|| as to v
def fairPR(M, index, phi, k):
    p,Q = uniformPR(M)
    K = top_k(p, Q, k)
    index_top_k_1, index_top_k = new_index(p, index)#true if in protected category(1) and on top-k.
    n = p.size
    G = matrix([matrix(-1*np.eye(n)), matrix(K)])
    h = matrix([matrix(np.zeros(n)), matrix(np.zeros(k*(n-k)))]) # inequality constraint seems correct
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

def check_solution(index, p, fp, k):
    n = len(p)
    sorted_index = np.argsort(-p)
    fp = np.array(fp)
    fp = fp.flatten()
    fr = np.array([0.,0.])
    for i in sorted_index[:k]:
        if index[i]:
            fr[1] += fp[int(i)]
        else:
            fr[0] += fp[int(i)]
    fr = fr/fr.sum()
    print("fractions: ", fr)
    sum_fp = 0.
    for i in p:
        sum_fp += float(i)
    print(sum_fp)

if len(sys.argv) != 3:#---------------------------------------- make independet from phi-pr as argument. Calculate from index vector.
    print("provide 2 arguments <ratio for protected category>, <top-k>")
else:
    pr = float(sys.argv[1])
    k = int(sys.argv[2])


M, index = create_adj_matrix()

if pr == 0:
    pr = sum(index) / len(index)
print("phi: ", pr)

j, fp = fairPR(M,index, pr, k)

j = np.array(j).flatten()
fp = np.array(fp).flatten()

with open("out_sensitive_topk_jump_v.txt", "w") as file_one:
    for i in j:
        file_one.write(str(i) + "\n")
with open("out_sensitive_topk_pagerank.txt", "w") as file_one:
    for i in fp:
        file_one.write(str(i) + "\n")

'''
# TODO: Change from matrix to numpy arrays before writing for better future parsing.
with open("opt_jump_vector_top%d.txt" %k, "w") as file_one:
    file_one.write("#jump vector\n%s" %str(j))
with open("opt_fairpagerank_vector_top%d.txt" %k, "w") as file_one:
    file_one.write("#fair pagerank vector\n%s" %str(fp))
'''


