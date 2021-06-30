import scipy
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from scipy.optimize import minimize
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.linalg import block_diag
from numpy import linalg as LA
from scipy.stats import norm
from utils import *
from torch.quasirandom import SobolEngine

class GP(object):
    def __init__(self, B, Noise=False, noise_delta=1e-4, verbose=0, compress=False):  # noise_delta = std (sigma)
        self.B = B
        self.dim = B.shape[0]
        self.verbose = verbose
        self.isCompressed = compress
        self.compress = lambda x: to_unit_cube(x, self.B[:, 0], self.B[:, 1]) if self.isCompressed and np.all(x >= self.B[:, 0]) and np.all(self.B[:, 1] >= x) else x
        
        assert Noise == False # for now
        if Noise == True:
            self.noise_delta = noise_delta
        else:
            self.noise_delta = 1e-4

        # reset GP
        self.clear()
        
    def clear(self):
        self.X = None
        self.y = None
        self._X = None
        self._y = None
        self.fitted = False
            
        self.hyper = {}
        self.hyper["var"] = 1
        self.hyper["lengthscale"] = 1
        
        self._B = self.compress(self.B)
        self.normalize = (0, 1)
    
    def _wrap(self):
        if self.isCompressed:
            self._X = self.compress(self.X)
            mu, sigma = np.median(self.y), self.y.std()
            sigma = 1.0 if sigma < 1e-6 else sigma
            self._y = (self.y - mu) / sigma
            self.normalize = (mu, sigma)
        else:
            self._X = self.X
            self._y = self.y
            
    def _unwrap(self, x):
        if self.isCompressed:
            return from_unit_cube(x, self.B[:, 0], self.B[:, 1])
        else:
            return x
        
    def _unnormal(self, y):
        m, v = self.normalize
        return m + v*y
    
    def _normal(self, y):
        m, v = self.normalize
        return (y - m)/v

    def set_data(self, X, y): # X: input 2d array [N*d], y: output 2d array [N*1]
        if X.ndim == 1:
            X = X.reshape((1, -1))
        assert X.shape[1] == self.dim
        self.X = X
        self.y = np.reshape(y, (self.X.shape[0], 1)) # the standardised output N(0,1)
        self._wrap()
        self.fitted = False

    def add_data(self, X, y): # X [N*d], y [N*1]
        assert len(y.shape) != 0
        if X.ndim == 1:
            X = X.reshape((1, -1))
        self.X = np.vstack((self.X, np.reshape(X, (X.shape[0], -1))))
        self.y = np.vstack((self.y, np.reshape(y, (y.shape[0], -1))))
        self._wrap()
        self.fitted = False

    def set_hyper(self, lengthscale=1, variance=1):
        self.hyper["lengthscale"] = lengthscale
        self.hyper["var"] = variance

    def get_hyper(self):
        return self.hyper

    def cov_RBF(self, x1, x2, hyper=None): # RBF Kernel
        if hyper == None:
            hyper = self.get_hyper()

        variance = hyper["var"]
        lengthscale = hyper["lengthscale"]
        
#         assert np.all(x1 >= self._B[:, 0]) and np.all(x1 <= self._B[:, 1])
#         assert np.all(x2 >= self._B[:, 0]) and np.all(x2 <= self._B[:, 1])

        assert x1.shape[1] == x2.shape[1] # 2d

        sqdist = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2,1) - 2*np.dot(x1, x2.T)
        return variance * np.exp(-0.5 * sqdist / np.square(lengthscale))

    def log_lik(self, hyper_values):
        # min the -ve loglk for the estimation of ls and var
        hyper = {}
        hyper["lengthscale"] = hyper_values[0]
        hyper["var"] = hyper_values[1]
        
        KK_x_x = self.cov_RBF(self._X, self._X, hyper) + np.eye(len(self._X)) * self.noise_delta**2
        if np.isnan(KK_x_x).any():  # NaN
            print("NaN in KK_x_x")
            raise ValueError("NaN in KK_x_x")
        
        try:
            L = scipy.linalg.cholesky(KK_x_x + 1e-6*np.eye(self._X.shape[0]), lower=True)
            temp = np.linalg.solve(L, self._y)
            alpha = np.linalg.solve(L.T, temp) # found the alpha for given hyper parameters
        except:
            return - np.inf

        log_lik = -1/2*np.dot(self._y.T, alpha) - np.sum(np.log(np.diag(L))) - 0.5*len(self._y)*np.log(2*np.pi)
        return np.asscalar(log_lik)

    def optimize(self): # Optimise the GP kernel hyperparameters
        opts = {"maxiter": 200, "maxfun": 200, "disp": False}
        bounds = np.asarray([[1e-3, 10] , [0.05, 1e5]])  # bounds on Lenghtscale and kernal Variance

        W = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(30, 2))
        loglik = np.array([])

        for x in W:
            loglik = np.append(loglik, self.log_lik(hyper_values=x))
        
        x0 = W[np.argmax(loglik)]
        Res = minimize(lambda x: - self.log_lik(hyper_values=x), 
                                x0,
                                bounds=bounds, 
                                method="L-BFGS-B", 
                                options=opts) # L-BFGS-B

        if self.verbose:
            print("estimated lengthscale and variance = ", Res.x)
        
        ls, var = Res.x
        self.set_hyper(ls, var) # update hyper 
        return Res.x
    
    def _resize(self, Xt):
        # assume self.X defined
        if len(Xt.shape) == 1:  # 1d
            Xt = np.reshape(Xt, (-1, self.X.shape[1]))
        
        if Xt.shape[1] != self.X.shape[1]:  # match dimension
            Xt = np.reshape(Xt, (-1, self.X.shape[1]))
        return Xt
    
    def prior(self, Xt): # Xt: the testing points  [M*d]
        M = Xt.shape[0]
        _Xt = self.compress(self._resize(Xt))
        meanPrior = np.zeros((M, 1))
        covPrior = self.cov_RBF(_Xt, _Xt, self.get_hyper())
        return meanPrior, covPrior

    def fit(self): # find alpha with self.hyper
        # self.K represents Ky (with noise)
        self.K = self.cov_RBF(self._X, self._X, self.hyper) + np.eye(len(self._X)) * (self.noise_delta**2)
        if np.isnan(self.K).any():  # NaN
            print("NaN in _K")
            
        try:
            self.L = scipy.linalg.cholesky(self.K, lower=True)
        except:
            self.L = scipy.linalg.cholesky(self.K + 1e-6*np.eye(self._X.shape[0]), lower=True)
        temp = np.linalg.solve(self.L, self._y)
        self.alpha = np.linalg.solve(self.L.T, temp) # algorithm 15.1
        self.fitted = True
        return self.alpha

    def posterior(self, Xt): # Xt: the testing points [M*d]
        assert self.fitted == True
        _Xt = self.compress(self._resize(Xt)) # reshape into [M*d]
        KK_xT_xT = self.cov_RBF(_Xt, _Xt, self.hyper)
        KK_x_xT = self.cov_RBF(self._X, _Xt, self.hyper)

        meanPost = np.reshape(np.dot(KK_x_xT.T, self.alpha), (-1, 1))
        v = np.linalg.solve(self.L, KK_x_xT)
        covPost = KK_xT_xT - np.dot(v.T, v)
        # var = np.reshape(np.diag(var), (-1, 1))
        return meanPost, covPost

    def ucb(self, x, b): # ucb at one point x, b for hyper, x:: 1*d
        x = x.reshape(1, -1)
#         assert np.all(x >= self.B[:, 0]) and np.all(x <= self.B[:, 1])
        
        mu, covar = self.posterior(x)
        mu = np.squeeze(mu)
        s = np.sqrt(np.diag(covar))
        return np.array(mu.reshape(-1, 1) - np.sqrt(b)*s.reshape(-1, 1)).item()

    def ucb_minimize(self, b): # fitted gp, bound, hyperparam b
        mesh = np.random.uniform(self.B[:, 0], self.B[:, 1], size=(1000, self.B.shape[0]))
        fx = np.array([self.ucb(x, b) for x in mesh])
        x0 = mesh[np.argmin(fx)]
        Res = minimize(lambda x: self.ucb(x, b), 
                                x0,
                                bounds=self.B, 
                                method="L-BFGS-B") # L-BFGS-B
        return Res.x, self.ucb(Res.x, b)
    
    def PI(self, x, y_best): # given one point, x
        x = x.reshape(1, -1)
        assert np.all(x >= self.B[:, 0]) and np.all(x <= self.B[:, 1])
        
        mu, covar = self.posterior(x)
        mu = np.squeeze(mu).item()
        s = np.sqrt(np.diag(covar)).item()
        if s == 0:
            z = np.sign(y_best - mu) * np.inf            
        else:
            z = (y_best - mu)/s
            
        prob = norm.cdf(z)
        if prob == np.nan:
            raise ValueError('Nan for PI')
        return prob
    
    def PI_maximize(self, y_best):
        mesh = np.random.uniform(self.B[:, 0], self.B[:, 1], size=(1000, self.B.shape[0]))
        fx = np.array([-self.PI(x, y_best) for x in mesh])
        x0 = mesh[np.argmin(fx)]
        Res = minimize(lambda x: -self.PI(x, y_best), 
                                x0,
                                bounds=self.B, 
                                method="L-BFGS-B")
        return Res.x, self.PI(Res.x, y_best)
    
    def EI(self, x, y_best):
        x = x.reshape(1, -1)
        assert np.all(x >= self.B[:, 0]) and np.all(x <= self.B[:, 1])
        
        mu, covar = self.posterior(x)
        mu = np.squeeze(mu).item()
        s = np.sqrt(np.diag(covar)).item()
        
        if s == 0:
            z = np.sign(y_best - mu) * np.inf            
        else:
            z = (y_best - mu)/s
            
        return -(z * norm.cdf(z) +  norm.pdf(z)) * s
    
    def EI_minimize(self, y_best): # fitted gp, bound, hyperparam b
        mesh = np.random.uniform(self.B[:, 0], self.B[:, 1], size=(1000, self.B.shape[0]))
        fx = np.array([self.EI(x, y_best) for x in mesh])
        x0 = mesh[np.argmin(fx)]
        Res = minimize(lambda x: self.EI(x, y_best), 
                                x0,
                                bounds=self.B, 
                                method="L-BFGS-B") # L-BFGS-B
        return Res.x, self.EI(Res.x, y_best)
    
    def Thompson_sample(self):
        seed = np.random.randint(int(1e6))
        sobol = SobolEngine(self.dim, scramble=True, seed=seed)
        Grid = from_unit_cube(np.array(sobol.draw(10000)), self.B[:, 0], self.B[:, 1])
        mu, covar = self.posterior(Grid)
        mu = np.squeeze(mu)
        s = np.sqrt(np.diag(covar))

        L = np.linalg.cholesky(covar + 1e-6*np.eye(10000)) # LL^T = Sigma (posterior covariance)
        f_post = mu.reshape(-1,1) + np.dot(L, np.random.normal(size=(10000, 1)))
        arg_min = np.argmin(f_post)
        w = Grid[arg_min]
        return w
    