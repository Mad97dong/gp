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
class localGD:
    def __init__(self, gp, B, w, f):
        self.gp = gp
        self.B = B
        self.lb = B[:, 0]
        self.ub = B[:, 1]
        
        self.w = w # tracing GD from w
        self.f = f # query function
        self.min = f(w)
        
        self.history_w = [w]
        self.history_full_w = [w] # for momentum accelerated
        self.FAIL = 0
        self.STOP = 4

        
    def _grad(self):
        return self.gp.grad(self.w)
    
    def update(self, commit=False):
        assert self.FAIL < self.STOP
        skip = 25
        clipnorm = 1
        
        implicit_path = []
        w = self.w
        block = np.ones(self.gp.dim)
        
        for i in range(skip):
            gt = block*self.gp.grad_sample(w)
            if LA.norm(gt) > clipnorm:
                gt = gt*clipnorm / LA.norm(gt)
                
            if LA.norm(gt) < 1e-4*self.gp.dim:
                self.STOP = 0 # break out
                print("CAVEAT: negligible norm")
                
            lr = self.line(w, gt)
            if lr == 0:
                print('lr = 0')
                break

            w_next = np.clip(w - lr*gt, self.lb, self.ub)
            w = w_next
            implicit_path.append(w)
            
#             m, v = self.gp.posterior_full_grad('full', self.gp.dim*[w])
#             variance = self.gp.get_hyper()["var"]
#             lengthscale = self.gp.get_hyper()["lengthscale"]

            
            # Coordinate
#             block = np.sqrt(variance / lengthscale**2) * 0.7 > np.squeeze(np.sqrt(np.diag(v)))
#             if all([c == False for c in block]):
#                 break 

            # Confidence Interval
#             pr = np.mean( np.sqrt(variance/ lengthscale**2) * 0.7 > np.squeeze(np.sqrt(np.diag(v))))
#             if np.random.uniform(0, 1, 1).item() > pr:
#                 break 
                
#         print('GD update: ', i)
        if commit == True:
            self.commit(w, implicit_path)
            
        return w, implicit_path
    
    def commit(self, w, path): # new query w, path arrive at w
        delta = np.abs(5e-3 * self.min)
        assert (w == path[-1]).all()
        assert self.min == self.f(self.w)
        
        if self.f(w) < self.min - delta:
            self.FAIL = 0
        else:
            self.FAIL += 1
            
        if self.f(w) < self.min:
            self.w = w
            self.min = self.f(w)

        self.history_w.append(w)
        self.history_full_w += path
        
        if len(self.history_full_w) > 51:
            self.STOP = 1

    # line search via ucb
    def line(self, w, gt, y_best, min_lr=0.005, max_lr=0.05):
        def subspace(lr):
            return self.gp.ucb(np.clip(w - lr*gt, self.lb, self.ub), y_best)
        
        lr_cand = np.linspace(min_lr, max_lr, 30)
        Score = np.array([subspace(lr) for lr in lr_cand])
        lr = lr_cand[np.argmin(Score)]
        
        Res = minimize(subspace, lr, bounds=[(min_lr, max_lr)], method="L-BFGS-B")
        return Res.x.item()
    