from sklearn.metrics.pairwise import pairwise_distances
import time
import pandas as pd
import sys
import tensorflow as tf
import numpy as np
from numba import cuda
import numba



USE_64 = False

if USE_64:
    bits = 64
    np_type = np.float64
else:
    bits = 32
    np_type = np.float32

@cuda.jit("void(float{}[:, :], float{}[:, :])".format(bits, bits))
def distance_matrix(mat, out):
    m = mat.shape[0]
    n = mat.shape[1]
    i, j = cuda.grid(2)
    d = 0
    if i < m and j < m:
        for k in range(n):
            tmp = mat[i, k] - mat[j, k]
            d += tmp * tmp
        out[i, j] = d**0.5 #should take square root

def gpu_dist_matrix(mat):
    rows = mat.shape[0]

    block_dim = (16, 16)
    grid_dim = (int(rows/block_dim[0] + 1), int(rows/block_dim[1] + 1))

    stream = cuda.stream()
    mat2 = cuda.to_device(np.asarray(mat, dtype=np_type), stream=stream)
    #print("mat2 done")
    out2 = cuda.device_array((rows, rows))
    #print("out2 done")
    distance_matrix[grid_dim, block_dim](mat2, out2)
    #print("distance done")
    out = out2.copy_to_host(stream=stream)
    #print("out done")

    return out


@numba.jit(parallel=True, fastmath=True)
def parallel_dist_matrix(mat):
    mat = mat.astype(np_type)
    n = mat.shape[0]
    m = mat.shape[1]
    result = np.zeros([n,n], np_type)
    for i in range(n):
        for j in range(n):
            temp = mat[i] - mat[j]
            result[i,j] = np.dot(temp,temp)

    return result


#tensorflow version
def dist_matrix_tf(points):
    
    u'''
    if expanded, is equal to:
    ptf = tf.placeholder("double")
    expd=tf.expand_dims(ptf,2) # from (n_emb x emb_dim) to (n_emb x emb_dim x 1)
    tiled=tf.tile(expd, [1,1,tf.shape(ptf)[0]]) # copying the same matrix n times
    trans=tf.transpose(ptf)
    num=tf.reduce_sum(tf.squared_difference(trans,tiled), 1)
    den1=1-tf.reduce_sum(tf.square(ptf),1)
    den1=tf.expand_dims(den1, 1)
    den=tf.matmul(den1, tf.transpose(den1))
    tot=1+2*tf.div(num, den)
    '''
    ptf = tf.placeholder("double")
    tot=1+2*tf.div(tf.reduce_sum(tf.squared_difference(tf.transpose(ptf), tf.tile(tf.expand_dims(ptf,2), [1,1,tf.shape(ptf)[0]])), 1), tf.matmul(tf.expand_dims(1-tf.reduce_sum(tf.square(ptf),1), 1), tf.transpose(tf.expand_dims(1-tf.reduce_sum(tf.square(ptf),1), 1))))
    sess = tf.Session()
    return sess.run(tot, feed_dict={ptf: points})


def dist_matrix_tf_all(X):
    with tf.device('/gpu:0'):
        #ptf=tf.Variable(tf.random_uniform([nvecs, 2], minval=-0.99, maxval=0.99), name="weights")
        ptf = tf.get_variable("weights", initializer=X)
        tot=1+2*tf.div(tf.reduce_sum(tf.squared_difference(tf.transpose(ptf), tf.tile(tf.expand_dims(ptf,2), [1,1,tf.shape(ptf)[0]])), 1), tf.matmul(tf.expand_dims(1-tf.reduce_sum(tf.square(ptf),1), 1), tf.transpose(tf.expand_dims(1-tf.reduce_sum(tf.square(ptf),1), 1))))
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    return (sess.run(tot))



def pairwise_dist (A, B):  
    '''
    Computes pairwise distances between each elements of A and each elements of B.
    Args:
        A,    [m,d] matrix
        B,    [n,d] matrix
    Returns:
        D,    [m,n] matrix of pairwise distances
    '''
    with tf.variable_scope('pairwise_dist'):
    # squared norms of each row in A and B
        na = tf.reduce_sum(tf.square(A), 1)
        nb = tf.reduce_sum(tf.square(B), 1)
    # na as a row and nb as a co"lumn vectors
        na = tf.reshape(na, [-1, 1])
        nb = tf.reshape(nb, [1, -1])
    # return pairwise euclidead difference matrix
        D = tf.sqrt(tf.maximum(na - 2*tf.matmul(A, B, False, True) + nb, 0.0))
    return D






data = pd.read_csv("../data/mnist_train.txt", header=None)
data = data.values
data = data[:int(sys.argv[1]), :-1]


'''
start = time.time()
d3 = dist_matrix_tf_all(data)
#d3 = pairwise_dist(data,data)
print("tf: %.4f" %(time.time()-start))
'''
'''
start = time.time()
d4 = parallel_dist_matrix(data)
print("Parallel jit: %.4f" %(time.time()-start))
'''
start = time.time()
d2 = gpu_dist_matrix(data)
print("GPU: %.4f" %(time.time()-start))
'''
start = time.time()
d1 = pairwise_distances(X=data, metric="euclidean", n_jobs=-1)
print("CPU parallel: %.4f" %(time.time()-start))
'''

