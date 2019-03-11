'''
need to carefully sample data from original as the anchors
try maintain the ratio of different classes
if one class has only one sample there is no way to get good classification
'''
import sys
import numpy as np
import pandas as pd
#from sklearn.metrics.pairwise import pairwise_distances#fastest, parallel
#from scipy.spatial.distance import cdist 
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
#from scipy.optimize import nnls
from collections import Counter
import time
sys.path.append("../model")
from hpc import *



def MSTKNN(dis, k): 
    start = time.time()
    d = csr_matrix(dis.astype('float'))
    Tcsr = minimum_spanning_tree(d)
    del d
    mpn = Tcsr.toarray().astype(float)
    del Tcsr
    mpn[mpn != 0] = 1 
    #print("MST time: %.4f" %(time.time()-start))
    
    start=time.time()
    for i in range(mpn.shape[0]):
        #index = np.argpartition(dis[i], k+1)[:(k+1)].tolist()#christmas
        #index.remove(i)#remove self #wrong, fix it!
        index = np.argsort(dis[i])[1:(k+1)]
        degree = mpn[i].sum()
        j = 0
        while degree < k:
            if mpn[i,index[j]] == 0:
                mpn[i, index[j]] = 1
                degree += 1
            j += 1
    #print("KNN time:%.4f" %(time.time()-start))
    mpn = mpn.astype(int)

    return mpn



def KNNgraph(dis, k):
    n = dis.shape[0] #KNN 1, nonKNN 0
    arr = np.zeros([n,n])
    for i in range(n):
        index = np.argsort(dis[i])[1:(k+1)] #0 itself
        arr[i,index] = 1

    return arr


def NNLSW(aff, graph, epsilon=None, verbose=False):
    rank = graph.shape[0]
    W = np.zeros((rank,rank))
    for i in range(rank):
        vec = aff[i]#b vector in scipy documentation
        gvec = graph[i]
        indK = [j for j in range(rank) if gvec[j] == 1]
        k = len(indK)
        mat = aff[indK]#A matrix in scipy documentation
        w = nnls(mat.T, vec)[0]#return both weights and residual
        if epsilon is not None:
            w = w + epsilon*max(w)#all neighbors nonzero
        if sum(w)==0:
            w = np.ones(len(w))
        w = w/sum(w) #need to normalize, w bounded between 0 and 1
        
        for ii in range(k):
            W[i, indK[ii]] = w[ii]
        if verbose and i%100 == 1:
            print("w", i)

    return W


def kNeighbors(mat, i, k): # return index of kNN based on L2
    nrow = mat.shape[0]
    center = mat[i]
    dis = [sum((mat[j]-center)**2) for j in range(nrow)]
    
    return np.argsort(dis)[1:(k+1)] #0 is self


def iterSolveQ(P, W, ita=1e-5, iteration=1000, verbose=False): 
    N = len(W)
    p = len(P)
    P = oneHot(P)
    Q = np.random.uniform(0,1,(N-p,P.shape[1]))
    PQ = np.vstack((P,Q))
    W = W[p:]
    err = np.inf
    step = 0
    while err > ita and step < iteration: #practically it converges fast, may add some analysis or proof
        dump = PQ[p:N].copy()
        PQ[p:N] = np.dot(W, PQ)
        err = sum(sum(abs(PQ[p:N]-dump))) / sum(sum(abs(dump)))
        if (verbose == True):
            print("%d %f" %(step, err))
            step += 1
    if step >= iteration:
        print("Warning: non-convergence with %d iterations." %step)

    return PQ



def oneHot(y, verbose=False):
    label = list(set(y.flatten()))
    dic = {}
    for l in label:
        dic[l] = label.index(l)
    if verbose:
        print(dic)
    L = len(label)
    n = len(y)
    encoding = np.zeros((n,L))
    for i in range(n):
        encoding[i,dic[y[i]]] = 1

    return encoding


def makeData(data, omega=0.1, maxi=None, shuffle=False, balance=False):
    '''
    randomly mask 1-omega sample labels
    '''
    if shuffle:#shuffle data
        index = list(range(data.shape[0]))
        np.random.shuffle(index)
        data = data[index]
    if maxi is not None:#take a subset of data
        num = min(len(data),maxi)
        data = data[:num]
        #print("Train-test total sample: %d" %num)

    n = len(data)
    n_kn = int(n*omega)
    X = data[:,:-1] #deafult last column is label
    y = data[:,-1]

    index = list(range(n))
    if balance:
        C = set(y)
        ind_known = []
        sample = int(n_kn/len(C))
        for c in C:
            ind_c = [i for i in index if y[i]==c]
            ind_c_select = np.random.choice(ind_c,min(len(ind_c),sample), replace=False)#make sure selected smaller than total for a specific class
            ind_known.extend(ind_c_select.tolist())
    else:
        ind_known = np.random.choice(index, n_kn, replace=False).tolist()
    ind_mask = list(set(index)-set(ind_known))
    index = ind_known + ind_mask

    y_true = y[ind_mask]
    y_p = y[ind_known]
    X = X[index]
    #print(Counter(y_p))
    
    return X, y_p, y_true


def evaluate(y_hat, y_true):
    y_hat = np.argmax(y_hat, axis=1)
    y_true = np.argmax(y_true, axis=1)

    return sum(y_hat==y_true)/len(y_hat)


def TC(X, y_p, delta, eta, verbose=False):
    start = time.time()
    #dis = cdist(X, X, metric="euclidean")
    #dis = pairwise_distances(X=X, metric='euclidean', n_jobs=-1)
    dis = gpu_dist_matrix(X)#get memory error when dataset is large
    #print("Dist time: %.3f" %(time.time()-start))
    p = len(y_p)
    start = time.time()
    #graph = MSTKNN(dis, delta)
    graph = fast_MSTKNN(dis, p, delta, n_jobs=-1)
    #print("Graph time: %.3f" %(time.time()-start))
    del dis
    start = time.time()
    W = multicore_nnls(nnlsw, X, graph, p, n_jobs=-1, epsilon=1e-2)
    #W = NNLSW(X, graph, epsilon=1e-2, verbose=verbose)
    #print("Weight time: %.3f" %(time.time()-start))
    del graph
    PQ = iterSolveQ(y_p, W, ita=eta, verbose=verbose)
    y_hat = PQ[len(y_p):]

    return y_hat



if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Specify fname, omega, delta and eta")
        exit(0)

    fname = sys.argv[1]#data file name
    omega = float(sys.argv[2])#labled sample rate
    delta = int(sys.argv[3])#minimum degree of graph
    eta = float('1e-'+sys.argv[4])#stopping criteria

    data = pd.read_csv("../data/"+fname, header=None)
    data = data.values
    X, y_p, y_true = makeData(data, omega=omega, maxi=5000,shuffle=True, balance=True)
    #X = X/255.#normalize
    
    start = time.time()
    dis = cdist(X, X, metric="euclidean")
    print("distance time: %3f" %(time.time()-start))
    start = time.time()
    #graph = KNNgraph(dis, k)
    graph = MSTKNN(dis, delta)
    print("graph time: %3f" %(time.time()-start))
    #np.savetxt("graph.txt", graph, fmt='%d', delimiter=',')
    W = NNLSW(X, graph, epsilon=1e-2)
    PQ = iterSolveQ(y_p, W, ita=eta, verbose = False)
    y_hat = PQ[len(y_p):]

    y_true = oneHot(y_true)
    print(evaluate(y_hat, y_true))
    
