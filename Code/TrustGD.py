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
class TrustGD:
    def __init__(self, gp, TR_c, f, TR_l):
        self.gp = gp
        
        self.lb = np.maximum(TR_c - TR_l/2, gp.B[:, 0])
        self.ub = np.minimum(TR_c + TR_l/2, gp.B[:, 1]) 
        self.B = np.vstack([self.lb, self.ub]).T
        
        self.xc = xc # tracing GD from xc
        self.history = []
        self.f = f # query function
        self.min = f(xc)
        
        self.TR_l = TR_l
        
    def update(self, commit=False):
        xc = self.xc
        intermediate = [xc]
        
        gt = self.gp.grad_sample(xc)
        gt = gt / LA.norm(gt)
        list_gt = []
        
        for i in range(3):
            dt = np.random.normal(size=self.gp.dim)
            dt = dt / LA.norm(dt)
            list_gt.append(gt * dt)

        lr, gt = self.pick_line(xc, list_gt)
        xc = np.clip(xc - lr*gt, self.lb, self.ub)
        
        lr = (1/2)**10 * self.TR_l
        new_xc = np.clip(xc - lr*gt, self.lb, self.ub)
        
        print('prev = ', self.f(xc))
        count = 0
        while self.gp.ucb(xc, 1) >= self.gp.ucb(new_xc, 1):
            xc = new_xc
            intermediate.append(xc)
            gt = self.gp.grad_sample(xc)
            gt = gt / LA.norm(gt)
            new_xc = np.clip(xc - lr*gt, self.lb, self.ub)
            count += 1
            
        print('GD = ', self.f(xc))
        if commit == True:
            self.commit(xc, intermediate)
            
        return xc, intermediate

#     def pick_line(self, xc, list_gt):
#         list_lr = [self.line(xc, gt) for gt in list_gt]
#         list_xc = [np.clip(xc - lr*gt, self.lb, self.ub) for (lr, gt) in zip(list_lr, list_gt)]
#         list_Score = [self.gp.ucb(xc_cand, 10) for xc_cand in list_xc]
#         index = np.argmin(list_Score)
#         return list_lr[index], list_gt[index]
    
#     def line(self, xc, gt):
#         min_lr = 1e-1
#         max_lr = self.TR_l
        
#         def subspace(lr):
#             return self.gp.ucb(np.clip(xc - lr*gt, self.lb, self.ub), 10)
        
#         lr_cand = np.linspace(min_lr, max_lr, 30)
#         Score = np.array([subspace(lr) for lr in lr_cand])
        
#         index = np.random.choice(np.where(Score == Score.min())[0])
#         lr = lr_cand[index]        
#         Res = minimize(subspace, lr, bounds=[(min_lr, max_lr)], method="L-BFGS-B")
#         return Res.x.item()
    
    
    def commit(self, xc, intermediate): # new query w, path arrive at w
        delta = np.abs(1e-3 * self.min)
        assert (xc == intermediate[-1]).all()
        assert self.min == self.f(self.xc)
            
        self.xc = xc
        self.history += intermediate
        self.min = self.f(xc)