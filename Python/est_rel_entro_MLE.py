import math
from numpy import array
import numpy as np


def est_rel_entro_MLE(sampP, sampQ):
    """MLE estimate of KL divergence D(P||Q) in bits of input sample.

    This function returns a scalar MLE of the KL divergence D(P||Q) between
    sampP and samp Q when they are vectors, or returns a row vector
    consisting of the MLE of each column of the sample when they are
    matrices.

    Input:
    ----- sampP: a vector or matrix which can only contain integers. The
                 input data type can be any integer types such as uint8/int8/
                 uint16/int16/uint32/int32/uint64/int64, or floating-point
                 such as single/double.
    ----- sampQ: same conditions as sampP. Must have the same number of
                 columns if a matrix.
    Output:
    ----- the KL divergence (in bits) of the input vectors or that of
               each column of the input matrices. The output data type is
               double.
    """
    sampP = formalize_sample(sampP)
    sampQ = formalize_sample(sampQ)
    
    [m, sizeP] = sampP.shape
    [n, seq_num] = sampQ.shape
    n = float(n)
    m = float(m)

    if (sizeP != seq_num):
        raise Exception('Input arguments P and Q must have the same number '
                        'of columns')

#  empirical distros + fingerprints
#
#  Map non-consecutive integer samples to consecutive integer numbers
#  along each column of X and Y (which start with 1 and end with the
#  total number of distinct samples in each column). For example,
#                  [  1    6    4  ]        [ 1  3  3 ]
#                  [  2    6    3  ] -----> [ 2  3  2 ]
#                  [  3    2    2  ]        [ 3  1  1 ]
#                  [ 1e5   3   100 ]        [ 4  2  4 ]
#  The purpose of this mapping is to shrink the effective data range to
#  avoid possible numerical overflows.

    concat = np.vstack([sampP, sampQ])
    [PQ_len, PQ_wid] = concat.shape
    [PQ_seq, dex] = [np.sort(concat, axis=0), np.argsort(concat, axis=0)]
    
    rows = np.mod(dex + np.arange(PQ_wid)*PQ_len, PQ_len)
    cols = np.tile(np.arange(PQ_wid), (PQ_len, 1))
    vals = np.cumsum(np.vstack([np.ones((1, PQ_wid), dtype=np.int64),
                                np.sign(np.diff(PQ_seq, axis=0))]), axis=0)
    PQ_seq[rows, cols] = vals.reshape(PQ_seq[rows, cols].shape)
    S = np.amax(PQ_seq)
    sampP = PQ_seq[:int(m)]
    sampQ = PQ_seq[int(m):]
    
    e_p = np.apply_along_axis(lambda x: np.bincount(x - 1, minlength=S),
                              axis=0, arr=sampP)
    e_q = np.apply_along_axis(lambda x: np.bincount(x - 1, minlength=S),
                              axis=0, arr=sampQ)
    bins = np.amax(np.hstack([e_p, e_q]))
    prob_q = np.arange(bins + 1)[:, np.newaxis] / n
    prob_mat = log_mat_MLE(prob_q, n, seq_num)

    sum_p = np.zeros(prob_mat.shape)
    for row_iter in np.nditer((np.unique(e_q))[:, np.newaxis]):
        sum_p[row_iter] = np.sum(e_p * (e_q == row_iter), axis=0) / m
    d = np.sum((sum_p * prob_mat), axis=0) / math.log(2)
    entro = est_entro_MLE(sampP)
    est = np.maximum(0, - entro - d)
    return est

def log_mat_MLE(x, n, seq_num):
    X = np.tile(x, [1, seq_num])
    X[X == 0] = 1.0 / n
    return np.log(X)

def est_entro_MLE(samp):
    """MLE estimate of Shannon entropy (in bits) of input sample

    This function returns a scalar MLE estimate of the entropy of samp when 
    samp is a vector, or returns a row vector containing the MLE estimate of
    each column of samp when samp is a matrix.

    Input:
    ----- samp: a vector or matrix which can only contain integers. The
                 input data type can be any integer types such as uint8/int8/
                 uint16/int16/uint32/int32/uint64/int64, or floating-point
                 such as single/double.
    Output:
    ----- est: the entropy (in bits) of the input vector or that of each
               column of the input matrix. The output data type is double.
    """
    
    samp = formalize_sample(samp)
    [n, wid] = samp.shape
    n = float(n)

    f = fingerprint(samp)
    prob = np.arange(1, f.shape[0] + 1) / n
    prob_mat = - prob * np.log2(prob)
    return prob_mat.dot(f)

def formalize_sample(samp):
    samp = np.array(samp)
    if np.any(samp != np.fix(samp)):
        raise ValueError('Input sample must only contain integers.')
    if samp.ndim == 1 or samp.ndim == 2 and samp.shape[0] == 1:
        samp = samp.reshape((samp.size, 1))
    return samp
    
def fingerprint(samp):
    """A memory-efficient algorithm for computing fingerprint when wid is
    large, e.g., wid = 100
    """
    wid = samp.shape[1]

    d = np.r_[
        np.full((1, wid), True, dtype=bool),
        np.diff(np.sort(samp, axis=0), 1, 0) != 0,
        np.full((1, wid), True, dtype=bool)
    ]

    f_col = []
    f_max = 0

    for k in range(wid):
        a = np.diff(np.flatnonzero(d[:, k]))
        a_max = a.max()
        hist, _ = np.histogram(a, bins=a_max, range=(1, a_max + 1))
        f_col.append(hist)
        if a_max > f_max:
            f_max = a_max

    return np.array([np.r_[col, [0] * (f_max - len(col))] for col in f_col]).T
