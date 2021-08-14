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
    def __init__(self, B, Noise=False, noise_delta=1e-4, p=0, verbose=0, compress=False):
        super().__init__(B, Noise, noise_delta, verbose, compress)
        self.p = p  # partial derivative along which the grad GP (0 to N-1)

    def set_p(self, p=0):
        self.p = p
    
    def K11(self, Xt, hyper=None):
        if hyper == None:
            hyper = self.get_hyper()
        
        v = hyper["var"]
        l = hyper["lengthscale"]
        
        Xt = self._shape(Xt)
#         assert np.all(Xt >= self._B[:, 0]) and np.all(Xt <= self._B[:, 1])
        Xt_p = np.reshape(Xt[:, self.p], (-1, 1))

        sqdist = sq_dist(Xt, Xt)
        sqdist_pp = sq_dist(Xt_p, Xt_p)
        return (v/(l**2)) * (1 - sqdist_pp / (l**2)) * np.exp(-0.5 * sqdist / (l**2))

    def K01(self, Xt, hyper=None):
        if hyper == None:
            hyper = self.get_hyper()

        v = hyper["var"]
        l = hyper["lengthscale"]
        
        Xt = self._shape(Xt)
#         assert np.all(Xt >= self._B[:, 0]) and np.all(Xt <= self._B[:, 1])
        
        Xt_p = Xt[:, self.p]
        _X_p = np.reshape(self._X[:, self.p], (-1, 1))

        sqdist = sq_dist(self._X, Xt)
        diff = _X_p - Xt_p
        K_01 = (v/(l**2)) * diff * np.exp(-0.5 * sqdist / (l**2))
        return K_01

    def Kpq(self, p, q, XPt, XQt, hyper=None):
        # covar between partial derivatives p and q (default: self.p, q)
        if p == q:
            raise ValueError('call K11 instead')
        
        if hyper == None:
            hyper = self.get_hyper()

        variance = hyper["var"]
        lengthscale = hyper["lengthscale"]
        
        XPt = self._shape(XPt)
        XQt = self._shape(XQt)
#         assert np.all(XPt >= self._B[:, 0]) and np.all(XPt <= self._B[:, 1])
#         assert np.all(XQt >= self._B[:, 0]) and np.all(XQt <= self._B[:, 1])
        
        XP_p = np.reshape(XPt[:, p], (-1, 1))
        XP_q = np.reshape(XPt[:, q], (-1, 1))
        diff_p = XP_p - XQt[:, p]
        diff_q = XP_q - XQt[:, q]
        
        sqdist = sq_dist(XPt, XQt)
        return -(variance/(lengthscale**4)) * (diff_p * diff_q) * np.exp(-0.5 * sqdist / (lengthscale**2))
    
           
    def prior_joint_MVN(self, p, q, XPt, XQt, full=False): # give the prior joint distribution of gp, partial p & q
        N = self.X.shape[0]
        M = XPt.shape[0]
        S = XQt.shape[0]
        coord = self.p
        
        _XPt = self._shape(XPt)
        _XQt = self._shape(XQt)

        Ky = self.cov_RBF(self._X, self._X) + np.eye(len(self._X)) * (self.noise_delta**2)

        self.set_p(p)
        K_01 = self.K01(_XPt)
        K_10 = K_01.T
        K_11 = self.K11(_XPt)
        
        self.set_p(q)
        K_02 = self.K01(_XQt)
        K_20 = K_02.T
        K_22 = self.K11(_XQt)
        
        K_12 = self.Kpq(p, q, _XPt, _XQt)
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
        
        
    def prior_full_MVN(self, coord, Xt): # give the prior joint distribution of gp, coord = [0, 1, 2, ...]
        if coord == 'full':
            coord = list(np.arange(self.dim))
        assert len(coord) == len(Xt)
        p_ = self.p
        
        _Xt = [self._shape(data) for data in Xt]
        n = len(coord)
        
        # Ky
        Ky = self.cov_RBF(self._X, self._X) + np.eye(len(self._X)) * (self.noise_delta**2)
        
        # k
        K_0d = []
        for i in range(n):
            self.set_p(coord[i])
            K_0d.append(self.K01(_Xt[i]))
        k_ = np.hstack(K_0d)    
        
        ## K square
        K_ = []
        for i in range(n):
            self.set_p(coord[i])
            K_1d = []
            for j in range(n):
                if j == i:
                    K_1d.append(self.K11(_Xt[i]))
                else:
                    K_1d.append(self.Kpq(coord[i], coord[j], _Xt[i], _Xt[j]).T)
            K_.append(np.vstack(K_1d))
        K_ = np.hstack(K_)
        
        self.set_p(p_) # set the self.p back
        return k_, K_

    def prior_grad(self, Xt):
        _Xt = self._shape(Xt)
        
        M = _Xt.shape[0]
        K_11 = self.K11(_Xt, _Xt)
        return np.zeros((M, 1)), K_11

    def posterior_grad(self, Xt):
        """
        Xt: the testing points  [M*d]
        Returns: posterior mean, posterior var of grad GP
        """
        assert self.fitted == True
        _Xt = self._shape(Xt)
        self.alpha = self.fit()
        K_01 = self.K01(_Xt)

        meanPost = np.reshape(np.dot(K_01.T, self.alpha), (-1, 1))
        v = np.linalg.solve(self.L, K_01)
        covPost = self.K11(_Xt) - np.dot(v.T, v)
        return meanPost, covPost
    
    def posterior_joint_grad(self, p, q, XPt, XQt): 
        """
        Xt: the testing points  [M*d]
        Returns: posterior mean, posterior var of the full gradient (partial p & q) of GP
        """
        assert self.fitted == True
        XPt = self._shape(XPt)
        XQt = self._shape(XQt)
        
        K_01, K_02, K_11, K_22, K_12 = self.prior_joint_MVN(p, q, XPt, XQt) # compress in prior_joint_MVN
        k_ = np.hstack([K_01, K_02])
        K_ = np.vstack([np.hstack([K_11, K_12]), np.hstack([K_12.T, K_22])])

        # posterior mean
        meanPost = np.reshape(np.dot(k_.T, self.alpha), (-1, 1))
        # posterior covariance
        v = np.linalg.solve(self.L, k_)
        covPost = K_ - np.dot(v.T, v)
        if np.linalg.det(covPost) < 0:
            print('Cov: ', covPost)
            print('covariance not positive semidefinite')
#             raise ValueError('covariance not positive semidefinite')
        return meanPost, covPost
    
    def posterior_full_grad(self, coord, Xt): ## usage: gp.posterior_full_grad('full', dim*[w])
        assert self.fitted == True
        if coord == 'full':
            coord = list(np.arange(self.dim))
        
        assert len(coord) == len(Xt)
        Xt = [self._shape(x) for x in Xt]
        k_, K_ = self.prior_full_MVN(coord, Xt)

        # posterior mean
        meanPost = np.reshape(np.dot(k_.T, self.alpha), (-1, 1))
        # posterior covariance
        v = np.linalg.solve(self.L, k_)
        covPost = K_ - np.dot(v.T, v)
        if np.linalg.det(covPost) < 0:
            print('Cov not positive semidefinite: ', covPost)

#             raise ValueError('covariance not positive semidefinite')
        return meanPost, covPost
    
    def _unnormal_grad(self, g):
        if self.CompressToCube:
            m, v = self.get_normal()
            return v*g   # v*g / (self.B[:, 1] - self.B[:, 0])
        else:
            return g
     
    def _normal_grad(self, g):
        if self.CompressToCube:
            m, v = self.get_normal()
            return g/v     # g * (self.B[:, 1] - self.B[:, 0]) / v
        else:
            return g
    
    def grad(self, x): # given a fit gp, return the posterior mean of gradient at x
        assert self.fitted == True
        if x.ndim == 1:
            x = x.reshape((1, -1))
        assert x.shape[1] == self.dim
        x = self._shape(x)
        
        p = self.p 
        
        M = np.zeros((x.shape[0], self.dim))
        for d in range(self.dim):
            self.set_p(d)
            md, vd = self.posterior_grad(x)
            md = np.reshape(md, (-1, 1))
            M[:, d] = md
            
        self.set_p(p)
        return M

    def grad_sample(self, x, option='full'):
        assert self.fitted == True
        if x.ndim == 1:
            x = x.reshape((1, -1))
        assert x.shape[1] == self.dim
        x = self._shape(x)
        
#         MM, VV = self.posterior_joint_grad(0, 1, x, x)
        MM, VV = self.posterior_full_grad('full', self.dim*[x])
        mean = np.squeeze(MM)
        covariance = np.squeeze(VV)
        if self.dim == 1:
            return np.random.normal(mean, covariance, 1)
        else:
            return np.random.multivariate_normal(mean, covariance, 1)

