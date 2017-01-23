from numpy import array
from est_rel_entro_HJW import *
from est_rel_entro_MLE import *
from rel_entropy_true import *

import math
import numpy as np
import matplotlib.pyplot as plt

def validate_dist(p):
    if np.imag(p).any() or np.isinf(p).any() or np.isnan(p).any() or (p < 0).any() or (p > 1).any():
        raise ValueError('The probability elements must be real numbers between 0 and 1.')

    eps = np.finfo(np.double).eps
    if (np.abs(p.sum(axis=0) - 1) > np.sqrt(eps)).any():
        raise ValueError('Sum of the probability elements must equal 1.')

def randsmpl(p, m, n):
    """Independent sampling from a discrete distribution

    x = randsmpl(p, m, n) returns an m-by-n matrix x with random samples
    drawn independently from the input (discrete) distribution specified
    with pmf p. Suppose that the sample space comprises K samples, then p
    must be a (row- or column-) vector containing K probability masses 
    summing to 1. The output, x(i,j) = k, k = 1, ..., K, describes that
    the k-th sample is drawn in the (i,j)-th trial, for i = 1, ..., m and
    j = 1,...,n. The default output data type of x is 'double'.

    The main idea is to divide interval [0,1] into K disjoint bins, 
    each with length proportional to the corresponding probability
    mass. Then, we draw samples from the uniform distribution U(0,1)  
    and determine the indices of the bins containing those samples. 

    Input:
    ----- p: discrete probability distribution summing to 1
    ----- m, n: the dimensions of the output matrix
    ----- dtype: the data type of the output, defaults to double
    Output:
    ----- an m-by-n matrix with random samples drawn from the input
          distribution p

    Peng Liu, Nov. 18, 2015
    """
    
    validate_dist(p)

    edges = np.r_[0, p.cumsum()]
    eps = np.finfo(np.double).eps
    if np.abs(edges[-1] - 1) > np.sqrt(eps):
        edges = edges / edges[-1]
    edges[-1] = 1 + eps

    return np.digitize(np.random.rand(m, n), edges)

if __name__ == '__main__':
    uS = 1 + 20*np.random.rand(1)
    cM = 1
    cN = 4
    num = 30
    mc_times = 20 # Total number of Monte-Carlo trials for each alphabet size
    record_S = np.ceil(np.logspace(1, 4, num))
    record_m = np.ceil(cM * record_S / np.log(record_S))
    record_n = np.ceil(cN * uS * record_S / np.log(record_S))
    record_true = np.zeros([num, mc_times])
    record_HJW = np.zeros([num, mc_times])
    record_MLE = np.zeros([num, mc_times])
    twonum = np.random.rand(2, 1)

    for i in range(num - 1, -1, -1):
        S = record_S[i]
        m = record_m[i]
        n = record_n[i]
        dist1 = np.random.beta(twonum[0], twonum[1], [int(S), 1])
        dist1 = dist1 / np.sum(dist1)
        ratio = 1 + (uS-1) * np.random.rand(*dist1.shape)
        dist2 = dist1 / ratio
        dist2[-1] = 1 - np.sum(dist2[0:-1])
        samp1 = randsmpl(dist1, int(m), mc_times)
        samp2 = randsmpl(dist2, int(n), mc_times)
        record_true[i, :] = np.tile(rel_entropy_true(dist1, dist2), [1, mc_times])
        record_HJW[i, :] = est_rel_entro_HJW(samp1, samp2)
        record_MLE[i, :] = est_rel_entro_MLE(samp1, samp2)

    HJW_err = np.sqrt(np.mean((record_HJW - record_true) ** 2, axis=1))
    MLE_err = np.sqrt(np.mean((record_MLE - record_true) ** 2, axis=1))

    plt.figure()
    plt.plot(record_S * uS / record_n, HJW_err, 'b-s', linewidth=2,
             markerfacecolor='b')
    plt.plot(record_S * uS / record_n, MLE_err, 'r-.o', linewidth=2,
             markerfacecolor='r')
    plt.xlabel('Su(S)/n')
    plt.ylabel('Root Mean Squared Error')
    plt.legend(['HJW', 'MLE'], loc='upper left')
    plt.title('Relative Entropy Estimation')
    plt.show()