from typing import overload
import scipy
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from numpy.linalg import inv
from GP import GP

def sq_dist(x1, x2): # x1: N*d, x2: M*d  --> N*d
    if x1.shape[1] != x2.shape[1]: # 2d
        x1 = np.reshape(x1, (-1, x2.shape[1]))
    return np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2*np.dot(x1, x2.T)

def diff(x1, x2): # x1: N*1, x2: M*1
    return x1 - np.squeeze(x2)

class GP_grad(GP):
    def __init__(self, SearchSpace, Noise=False, noise_delta=1e-4, p=0, verbose=0):
        super().__init__(SearchSpace, Noise, noise_delta, verbose)
        self.p = p  # partial derivative along which the grad GP (0 to N-1)

    def set_p(self, p=0):
        self.p = p

    def K11(self, Xtest, hyper=None): # RBF Kernel
        if hyper == None:
            hyper = self.get_hyper()

        variance = hyper["var"]
        lengthscale = hyper["lengthscale"]

        Xtest_p = np.reshape(Xtest[:, self.p], (-1, 1))

        sqdist = sq_dist(Xtest, Xtest)
        sqdist_pp = sq_dist(Xtest_p, Xtest_p)
        return (variance/(lengthscale**2)) * (1 - sqdist_pp / (lengthscale**2)) * np.exp(-0.5 * sqdist / (lengthscale**2))


    def K01(self, Xtest, hyper=None):
        if hyper == None:
            hyper = self.get_hyper()

        variance = hyper["var"]
        lengthscale = hyper["lengthscale"]

        Xtest_p = Xtest[:, self.p]
        X_p = np.reshape(self.X[:, self.p], (-1, 1))

        sqdist = sq_dist(self.X, Xtest)
#         print(sqdist.shape)
        diff = X_p - Xtest_p
        K_01 = (variance/(lengthscale**2)) * diff * np.exp(-0.5 * sqdist / (lengthscale**2))
        return K_01

    def Kpq(self, p, q, XPtest, XQtest, hyper=None):
        # covar between partial derivatives p and q (default: self.p, q)
        if p == q:
            raise ValueError('call K11 instead')
        
        if hyper == None:
            hyper = self.get_hyper()

        variance = hyper["var"]
        lengthscale = hyper["lengthscale"]
        
        XP_p = np.reshape(XPtest[:, p], (-1, 1))
        XP_q = np.reshape(XPtest[:, q], (-1, 1))
        diff_p = XP_p - XQtest[:, p]
        diff_q = XP_q - XQtest[:, q]
        
        sqdist = sq_dist(XPtest, XQtest)
   
        K_pq = (variance/(lengthscale**4)) * (diff_p * diff_q) * np.exp(-0.5 * sqdist / (lengthscale**2))
        return K_pq

    
#     def joint_MVN(self, Xtest): # give the joint distribution of gp and partial p
#         N = self.X.shape[0]
#         M = Xtest.shape[0]

#         if len(Xtest.shape) == 1:  # 1d
#             Xtest = np.reshape(Xtest, (-1, self.X.shape[1]))

#         K = self.cov_RBF(self.X, self.X)
#         K_11 = self.K11(Xtest)
#         K_01 = self.K01(Xtest)
#         K_10 = K_01.T
#         up = np.hstack([K, K_01])
#         down = np.hstack([K_10, K_11])
#         joint_cov = np.vstack([up, down])
#         return np.zeros((M+N, 1)), joint_cov

    def match_shape(self, Xtest):
#         if self.X == None:
#             raise ValueError('no self.X in GP')

        # assume self.X defined
        if len(Xtest.shape) == 1:  # 1d
            Xtest = np.reshape(Xtest, (-1, self.X.shape[1]))
        
        if Xtest.shape[1] != self.X.shape[1]:  # match dimension
            Xtest = np.reshape(Xtest, (-1, self.X.shape[1]))
        return Xtest
           
            
    def prior_joint_MVN(self, p, q, XPtest, XQtest, full=False): # give the prior joint distribution of gp, partial p & q
        N = self.X.shape[0]
        M = XPtest.shape[0]
        S = XQtest.shape[0]
        coord = self.p
        
        XPtest = self.match_shape(XPtest)
        XQtest = self.match_shape(XQtest)

        Ky = self.cov_RBF(self.X, self.X) + np.eye(len(self.X)) * (self.noise_delta**2)

        self.set_p(p)
        K_01 = self.K01(XPtest)
        K_10 = K_01.T
        K_11 = self.K11(XPtest)
        
        self.set_p(q)
        K_02 = self.K01(XQtest)
        K_20 = K_02.T
        K_22 = self.K11(XQtest)
        
        K_12 = self.Kpq(p, q, XPtest, XQtest)
        K_21 = K_12.T
        
        self.set_p(coord) # set the self.p back
        
        up = np.hstack([Ky, K_01, K_02])
        mid = np.hstack([K_10, K_11, K_12])
        down = np.hstack([K_20, K_21, K_22])
        joint_cov = np.vstack([up, mid, down])
#         return np.zeros((M+N+S, 1)), joint_cov
        if full == True:
            return joint_cov
        else:
            return K_01, K_02, K_11, K_22, K_12

    def prior_grad(self, Xtest):
        M = Xtest.shape[0]
        Xtest = self.match_shape(Xtest)
        K_11 = self.K11(Xtest, Xtest)
        return np.zeros((M, 1)), K_11


    def posterior_grad(self, Xtest):
        """
        Xtest: the testing points  [M*d]
        Returns: posterior mean, posterior var of grad GP
        """
        Xtest = self.match_shape(Xtest)
        self.alpha = self.fit()
        # assert self.K != None
        # assert self.L != None
        K_01 = self.K01(Xtest)

        meanPost = np.reshape(np.dot(K_01.T, self.alpha), (-1, 1))
        v = np.linalg.solve(self.L, K_01)
        covPost = self.K11(Xtest) - np.dot(v.T, v)
        # var = np.reshape(np.diag(var), (-1, 1))
        return meanPost, covPost
    
    def posterior_joint_grad(self, p, q, XPtest, XQtest): 
        """
        Xtest: the testing points  [M*d]
        Returns: posterior mean, posterior var of the full gradient (partial p & q) of GP
        """
        XPtest = self.match_shape(XPtest)
        XQtest = self.match_shape(XQtest)
        
        K_01, K_02, K_11, K_22, K_12 = self.prior_joint_MVN(p, q, XPtest, XQtest)
        k_ = np.hstack([K_01, K_02])
        K_ = np.vstack([np.hstack([K_11, K_12]), np.hstack([K_12.T, K_22])])
        
        self.alpha = self.fit()
        # posterior mean
        meanPost = np.reshape(np.dot(k_.T, self.alpha), (-1, 1))
        
        # posterior covariance
        v = np.linalg.solve(self.L, k_)
        covPost = K_ - np.dot(v.T, v)
        if np.linalg.det(covPost) < 0:
            print('Cov: ', covPost)
            print('XPtest', XPtest)
            print('XQtest', XQtest)
            raise ValueError('covariance not positive semidefinite')
        return meanPost, covPost
        
        