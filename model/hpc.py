import numpy as np
from numba import cuda
from scipy.optimize import nnls
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import multiprocessing as mp


USE_64 = False

if USE_64:
    bits = 64
    np_type = np.float64
else:
    bits = 32
    np_type = np.float32

@cuda.jit("void(float{}[:,:], float{}[:,:])".format(bits, bits))
def distance_matrix(mat, out):
    n = mat.shape[0]
    m = mat.shape[1]
    i, j = cuda.grid(2)
    d = 0 
    if i < n and j < n:
        for k in range(m):
            tmp = mat[i, k] - mat[j, k]
            d += tmp * tmp 
        out[i, j] = d**0.5 #should take square root 

def gpu_dist_matrix(mat):
    rows = mat.shape[0]
    block_dim = (16, 16) 
    grid_dim = (int(rows/block_dim[0] + 1), int(rows/block_dim[1] + 1)) 
   
    stream = cuda.stream()
    mat2 = cuda.to_device(np.asarray(mat, dtype=np_type), stream=stream)
    out2 = cuda.device_array((rows, rows))
    distance_matrix[grid_dim, block_dim](mat2, out2)
    out = out2.copy_to_host(stream=stream)

    return out 


def fast_MST(dis):
    d = csr_matrix(dis.astype('float'))
    Tcsr = minimum_spanning_tree(d)
    del d
    mpn = Tcsr.toarray().astype(float)
    del Tcsr
    mpn[mpn != 0] = 1
    mpn = mpn.astype(int)

    return mpn


def knn_base(mpn_raw, pid, sub_list, return_dic, dis, k):
    mpn = mpn_raw[sub_list].copy()
    for i in range(mpn.shape[0]):
        index = np.argsort(dis[sub_list[i]])[1:(k+1)]
        degree = mpn[i].sum()
        j = 0
        while degree < k:
            if mpn[i,index[j]] == 0:
                mpn[i,index[j]] = 1
                degree += 1
            j += 1

    return_dic[pid] = mpn

'''
def multicore_knn(func, mpn, dis, k, n_jobs=-1):
    total_cpu = mp.cpu_count()
    if type(n_jobs) is not int or n_jobs < -1 or n_jobs > total_cpu:
        print("Specify correct job number!")
        exit(0)
    elif n_jobs==-1:
        n_jobs = total_cpu

    mpn_list = np.array_split(range( mpn.shape[0]), n_jobs)#default axis=0
    processes = []
    return_dic = mp.Manager().dict()

    for i in range(n_jobs):
        proc = mp.Process(target=func, args=(mpn,i,mpn_list[i],return_dic,dis, k))
        processes.append(proc)
        proc.start()
    for process in processes:
        process.join()

    l = [return_dic[i] for i in range(n_jobs)]
        
    return np.concatenate(l, axis=0)
 
'''
def multicore_knn(func, mpn, dis, p, k, n_jobs=-1):
    total_cpu = mp.cpu_count()
    if type(n_jobs) is not int or n_jobs < -1 or n_jobs > total_cpu:
        print("Specify correct job number!")
        exit(0)
    elif n_jobs==-1:
        n_jobs = total_cpu

    mpn_list = np.array_split(range(p, mpn.shape[0]), n_jobs)#default axis=0
    processes = []
    return_dic = mp.Manager().dict()

    for i in range(n_jobs):
        proc = mp.Process(target=func, args=(mpn,i,mpn_list[i],return_dic,dis, k))
        processes.append(proc)
        proc.start()
    for process in processes:
        process.join()

    graph_p = mpn[:p].copy()
    graph_q = [return_dic[i] for i in range(n_jobs)]
    graph_q = np.concatenate(graph_q, axis=0)
    
    return np.concatenate([graph_p, graph_q], axis=0)


def fast_MSTKNN(dis, p, k, n_jobs=-1):
    mpn = fast_MST(dis)
    mpn = multicore_knn(knn_base, mpn, dis, p, k, n_jobs=n_jobs)

    return mpn


def nnlsw(aff, graph, pid, sub_list, return_dic, epsilon=None, verbose=False):
    nrows = len(sub_list)
    ncols = graph.shape[1]
    W = np.zeros((nrows, ncols))
    for i in range(nrows):
        ind_i = sub_list[i]
        vec = aff[ind_i]#b vector in scipy documentation
        gvec = graph[ind_i]
        indK = [j for j in range(ncols) if gvec[j] == 1]
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
    #print(W.shape)
    return_dic[pid] = W



'''
def multicore_nnls(func, aff, graph, n_jobs = -1, epsilon=None):
    total_cpu = mp.cpu_count()
    if type(n_jobs) is not int or n_jobs < -1 or n_jobs > total_cpu:
        print("Specify correct job number!")
        exit(0)
    elif n_jobs==-1:
        n_jobs = total_cpu

    graph_list = np.array_split(range( graph.shape[0]), n_jobs)#default axis=0
    processes = []
    return_dic = mp.Manager().dict()

    for i in range(n_jobs):
        proc = mp.Process(target=func, args=(aff,graph,i,graph_list[i],return_dic,epsilon))
        processes.append(proc)
        proc.start()
    for process in processes:
        process.join()
   
    l = [return_dic[i] for i in range(n_jobs)]

    return np.concatenate(l, axis=0)

'''
def multicore_nnls(func, aff, graph, p, n_jobs = -1, epsilon=None):
    total_cpu = mp.cpu_count()
    if type(n_jobs) is not int or n_jobs < -1 or n_jobs > total_cpu:
        print("Specify correct job number!")
        exit(0)
    elif n_jobs==-1:
        n_jobs = total_cpu

    graph_list = np.array_split(range(p, graph.shape[0]), n_jobs)#default axis=0
    processes = []
    return_dic = mp.Manager().dict()

    for i in range(n_jobs):
        proc = mp.Process(target=func, args=(aff,graph,i,graph_list[i],return_dic,epsilon))
        processes.append(proc)
        proc.start()
    for process in processes:
        process.join()
   
    W_p = np.eye(N=p, M=graph.shape[0])
    W_q = [return_dic[i] for i in range(n_jobs)]
    W_q = np.concatenate(W_q, axis=0)

    return np.concatenate([W_p,W_q], axis=0)
    
