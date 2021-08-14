import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d.axes3d import Axes3D

from PIL import Image
from scipy.optimize import minimize
from scipy import optimize
from GP import GP
from GP_grad import GP_grad
import time
import imageio
import functools

from torch.quasirandom import SobolEngine
import sobol_seq
from utils import *

# local search via GD
class lineGD:
    def __init__(self, gp, w, f, R):
        self.gp = gp
        self.B = gp.B
        self.lb = gp.B[:, 0]
        self.ub = gp.B[:, 1]
        
        self.w = w # tracing GD from w
        self.history = []
        self.f = f # query function
        self.min = f(w)
        
        self.R = R
        
    def update(self, commit=False):
        w = self.w
        
        Score = []
        gt = self.gp.grad_sample(w)
        gt = gt / LA.norm(gt)
        
        list_gt = []
        I = np.eye(self.gp.dim)
        for i in range(self.gp.dim):
            dt = I[i]
            list_gt.append(dt * gt)

        lr, gt = self.pick_line(w, list_gt)
        print('lr = ', lr)

        w = np.clip(w - lr*gt, self.lb, self.ub)

        intermediate = [w]
        if commit == True:
            self.commit(w, intermediate)
            
        return w, intermediate
    
    def commit(self, w, intermediate): # new query w, path arrive at w
        delta = np.abs(1e-3 * self.min)
        assert (w == intermediate[-1]).all()
        assert self.min == self.f(self.w)
            
        self.w = w
        self.history += intermediate
        self.min = self.f(w)

        
    def pick_line(self, w, list_gt):
        list_lr = [self.line(w, gt) for gt in list_gt]
        list_w = [np.clip(w - lr*gt, self.lb, self.ub) for (lr, gt) in zip(list_lr, list_gt)]
        list_Score = [self.gp.ucb(w_cand, 3) for w_cand in list_w]
        
        index = np.argmin(list_Score)
        return list_lr[index], list_gt[index]
    
    
    def line(self, w, gt):
        min_lr = 0 - self.R
        max_lr = self.R

        def subspace(lr):
            return self.gp.ucb(np.clip(w - lr*gt, self.lb, self.ub), 5)
        
        lr_cand = np.linspace(min_lr, max_lr, 30)
        Score = np.array([subspace(lr) for lr in lr_cand])
        
        index = np.random.choice(np.where(Score == Score.min())[0])
        lr = lr_cand[index]        
        Res = minimize(subspace, lr, bounds=[(min_lr, max_lr)], method="L-BFGS-B")
        return Res.x.item()

    
#     def line(self, w, gt):
#         min_lr = 1e-3
#         max_lr = self.R
        
#         seed = np.random.randint(int(1e6))
#         sobol = SobolEngine(1, scramble=True, seed=seed)

#         n_mesh = 1000
#         LR = from_unit_cube(np.array(sobol.draw(n_mesh)), np.array([min_lr]), np.array([max_lr]))
#         mesh = np.array([np.clip(w - lr*gt, self.lb, self.ub) for lr in LR])
        
#         mu, covar = self.gp.posterior(mesh)
#         mu = np.squeeze(mu)
#         s = np.sqrt(np.diag(covar))
        
#         L = np.linalg.cholesky(covar + 1e-6*np.eye(n_mesh)) # LL^T = Sigma (posterior covariance)
#         f_post = mu.reshape(-1,1) + np.dot(L, np.random.normal(size=(n_mesh, 1)))
#         index = np.random.choice(np.where(f_post == f_post.min())[0])
#         lr = LR[index]
#         return lr.item()
    

    
    

    