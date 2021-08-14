import numpy as np

def to_unit_cube(x, lb, ub):
    """Project to [0, 1]^d from hypercube with bounds lb and ub"""
    assert np.all(lb < ub) and lb.ndim == 1 and ub.ndim == 1 and x.ndim == 2
    xx = (x - lb) / (ub - lb)
    return xx


def from_unit_cube(x, lb, ub):
    """Project from [0, 1]^d to hypercube with bounds lb and ub"""
    assert np.all(lb < ub) and lb.ndim == 1 and ub.ndim == 1 and x.ndim == 2
    xx = x * (ub - lb) + lb
    return xx


def latin_hypercube(n_pts, dim):
    """Basic Latin hypercube implementation with center perturbation."""
    X = np.zeros((n_pts, dim))
    centers = (1.0 + 2.0 * np.arange(0.0, n_pts)) / float(2 * n_pts)
    for i in range(dim):  # Shuffle the center locataions for each dimension.
        X[:, i] = centers[np.random.permutation(n_pts)]

    # Add some perturbations within each box
    pert = np.random.uniform(-1.0, 1.0, (n_pts, dim)) / float(2 * n_pts)
    X += pert
    return X



def unique_row(X):
    """
    A functions to trim repeated rows that may appear when optimizing.
    :param X: array to trim repeated rows from
    :return: mask of unique rows
    """
    # Sort array and kep track of where things should go back to
    order = np.lexsort(X.T)
    reorder = np.argsort(order)

    X = X[order]
    diff = np.diff(X, axis=0)
    U = np.ones(len(X), 'bool')
    U[1:] = (diff != 0).any(axis=1)
    return U[reorder]


class close_data():
    def __init__(self, Xc, eps=1):
        self.Xc = Xc
        self.eps = eps
        
    # Iterable
    def __iter__(self):
        return self
    
    # Iterator
    def __next__(self):
        Xn = self.Xc + self.eps*np.random.uniform(-1, 1, self.Xc.shape)
        return Xn
    
    
def is_new(Xn, Xpast=None):
    if Xpast is None:
        return True
    else:
        return np.array([(i != Xn).any() for i in Xpast]).all()

def close_draw(Xc, n_random_draws=4, close=1, Xpast=None):
    # sample from a small 2d box with center given Xc, no repeated points
    X = np.zeros((n_random_draws, dim))
    X[0] = Xc
    counter = 0
    gc = close_data(Xc)
    while True:
        Xn = next(gc)
        if LA.norm(Xn - Xc) > 5e-2 and is_new(Xn, Xpast) == True:
            counter += 1
            X[counter] = Xn
        if counter == n_random_draws-1:
            break
    return X



def latin_hypercube(n_pts, dim):
    """Basic Latin hypercube implementation with center perturbation."""
    X = np.zeros((n_pts, dim))
    centers = (1.0 + 2.0 * np.arange(0.0, n_pts)) / float(2 * n_pts)
    for i in range(dim):  # Shuffle the center locataions for each dimension.
        X[:, i] = centers[np.random.permutation(n_pts)]

    # Add some perturbations within each box
    pert = np.random.uniform(-1.0, 1.0, (n_pts, dim)) / float(2 * n_pts)
    X += pert
    return X