import math

from numpy import array
import numpy as np


def rel_entropy_true(p, q):
    """KL divergence (relative entropy) D(p||q) in bits

    Returns a scalar entropy when the input distributions p and q are
    vectors of probability masses, or returns in a row vector the
    columnwise relative entropies of the input probability matrices p and
    q"""

    if type(p) == list or type(q) == tuple:
        p = np.array(p)
    if type(q) == list or type(q) == tuple:
        q = np.array(q)
        
    if not p.shape == q.shape:
        raise Exception('p and q must be equal sizes',
                        'p: ' + str(p.shape),
                        'q: ' + str(q.shape))

    if (np.iscomplex(p).any() or not
        np.isfinite(p).any() or
        (p < 0).any() or
        (p > 1).any()):
        raise Exception('The probability elements of p must be real numbers'
                        'between 0 and 1.')

    if (np.iscomplex(q).any() or not
        np.isfinite(q).any() or
        (q < 0).any() or
        (q > 1).any()):
        raise Exception('The probability elements of q must be real numbers'
                        'between 0 and 1.')

    eps = math.sqrt(np.spacing(1))
    if (np.abs(np.sum(p, axis=0) - 1) > eps).any():
        raise Exception('Sum of the probability elements of p must equal 1.')
    if (np.abs(np.sum(q, axis=0) - 1) > eps).any():
        raise Exception('Sum of the probability elements of q must equal 1.')

    return sum(np.log2((p**p) / (q**p)))
