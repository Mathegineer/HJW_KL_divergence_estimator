import math

from numpy import array

import numpy as np


def est_rel_entro_HJW(sampP, sampQ):
    """Estimate of KL divergence D(P||Q) in bits of input sample.

    This function returns a scalar estimate of the KL divergence D(P||Q)
    between sampP and samp Q when they are vectors, or returns a row vector
    consisting of the estimate of each column of the sample when they are
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
    [c_1, MLE_const] = const_gen(n)
    c_1 = np.tile(c_1, [1, seq_num])

    # order <= 21 to avoid floating point errors
    order = min(4 + math.ceil(1.2 * math.log(n)), 21)
    poly_entro = np.load('poly_coeff_entro.npy')
    coeff = -np.array(poly_entro[int(order)][1:])

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
    prob_mat = log_mat(prob_q, n, coeff, c_1, MLE_const)

    sum_p = np.zeros(prob_mat.shape)
    for row_iter in np.nditer((np.unique(e_q))[:, np.newaxis]):
        sum_p[row_iter] = np.sum(e_p * (e_q == row_iter), axis=0) / m
    d = np.sum((sum_p * prob_mat), axis=0) / math.log(2)
    entro = est_entro_JVHW(sampP)
    return np.maximum(0, - entro - d)

def log_mat(x, n, g_coeff, c_1, const):
    with np.errstate(divide='ignore', invalid='ignore'):
        K = g_coeff.shape[0] - 1
        thres = 2 * c_1 * math.log(n) / n
        [T, X] = np.meshgrid(thres, x)
        ratio = np.clip(2*X/T - 1, 0, 1)
        # force MATLAB-esque behavior with NaN, inf
        ratio[T == 0] = 1.0
        ratio[X == 0] = 0.0
        q = np.reshape(np.arange(K), [1, 1, K])
        g = np.tile(np.reshape(g_coeff, [1, 1, K + 1]), [c_1.shape[1], 1])
        g[:, :, 0] = g[:, :, 0] + np.log(thres)
        MLE = np.log(X) + (1-X) / (2*X*n)
        MLE[X == 0] = -np.log(n) - const
        tmp = (n*X[:,:,np.newaxis] - q)/(T[:,:,np.newaxis]*(n - q))
        polyApp = np.sum(np.cumprod(np.dstack([np.ones(T.shape + (1,)), tmp]),
                                    axis=2) * g, axis=2)
        polyFail = np.logical_or(np.isnan(polyApp), np.isinf(polyApp))
        polyApp[polyFail] = MLE[polyFail]
        return ratio*MLE + (1-ratio)*polyApp

def const_gen(n):
    const = 0
    c_1 = 0
    if math.log(n) < 3.2:
        const = 1
    elif math.log(n) < 3.4:
        const = 1.7
    elif math.log(n) < 4.9:
        const = 1.7
    elif math.log(n) < 6.9:
        const = 1.72
    elif math.log(n) < 7.2:
        c_1 = -0.340909 + 0.272727*math.log(n)
    elif math.log(n) < 9.3:
        c_1 = 2.49953 - 0.122084*math.log(n)
    elif math.log(n) < 11.4:
        c_1 = 2.10767 - 0.0718463*math.log(n)
    else:
        c_1 = 1.07
    return [c_1, const]


def est_entro_JVHW(samp):
    """Proposed JVHW estimate of Shannon entropy (in bits) of the input sample

    This function returns a scalar JVHW estimate of the entropy of samp when
    samp is a vector, or returns a row vector containing the JVHW estimate of
    each column of samp when samp is a matrix.

    Input:
    ----- samp: a vector or matrix which can only contain integers. The input
                data type can be any interger classes such as uint8/int8/
                uint16/int16/uint32/int32/uint64/int64, or floating-point
                such as single/double.
    Output:
    ----- est: the entropy (in bits) of the input vector or that of each column
               of the input matrix. The output data type is double.
    """
    samp = formalize_sample(samp)
    [n, wid] = samp.shape
    n = float(n)

    # The order of polynomial is no more than 22 because otherwise floating-point error occurs
    order = min(4 + int(np.ceil(1.2 * np.log(n))), 22)
    poly_entro = np.load('poly_coeff_entro.npy')
    coeff = np.array(poly_entro[int(order - 1)])

    f = fingerprint(samp)
    prob = np.arange(1, f.shape[0] + 1) / n

    # Piecewise linear/quadratic fit of c_1
    V1 = np.array([0.3303, 0.4679])
    V2 = np.array([-0.530556484842359, 1.09787328176926, 0.184831781602259])
    f1nonzero = f[0] > 0
    c_1 = np.zeros(wid)

    with np.errstate(divide='ignore', invalid='ignore'):
        if n >= order and f1nonzero.any():
            if n < 200:
                c_1[f1nonzero] = np.polyval(V1, np.log(n / f[0, f1nonzero]))
            else:
                n2f1_small = f1nonzero & (np.log(n / f[0]) <= 1.5)
                n2f1_large = f1nonzero & (np.log(n / f[0]) > 1.5)
                c_1[n2f1_small] = np.polyval(V2, np.log(n / f[0, n2f1_small]))
                c_1[n2f1_large] = np.polyval(V1, np.log(n / f[0, n2f1_large]))

            # make sure nonzero threshold is higher than 1/n
            c_1[f1nonzero] = np.maximum(c_1[f1nonzero], 1 / (1.9 * np.log(n)))

        prob_mat = entro_mat(prob, n, coeff, c_1)

    return np.sum(f * prob_mat, axis=0) / np.log(2)

def entro_mat(x, n, g_coeff, c_1):
    # g_coeff = {g0, g1, g2, ..., g_K}, K: the order of best polynomial approximation,
    K = len(g_coeff) - 1
    thres = 4 * c_1 * np.log(n) / n
    T, X = np.meshgrid(thres, x)
    ratio = np.minimum(np.maximum(2 * X / T - 1, 0), 1)
    q = np.arange(K).reshape((1, 1, K))
    g = g_coeff.reshape((1, 1, K + 1))
    MLE = - X * np.log(X) + 1 / (2 * n)
    polyApp = np.sum(np.concatenate((T[..., None], ((n * X)[..., None]  - q) / (T[..., None] * (n - q))), axis=2).cumprod(axis=2) * g, axis=2) - X * np.log(T)
    polyfail = np.isnan(polyApp) | np.isinf(polyApp)
    polyApp[polyfail] = MLE[polyfail]
    output = ratio * MLE + (1 - ratio) * polyApp
    return np.maximum(output, 0)

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