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
class GD_machine:
    def __init__(self, gp, w, f):
        self.gp = gp
        self.B = gp.B
        self.lb = gp.B[:, 0]
        self.ub = gp.B[:, 1]
        
        self.w = w # tracing GD from w
        self.f = f # query function
        self.min = f(w)
        
        self.history_w = [w]
        self.history_full_w = [w] # for momentum accelerated
        self.FAIL = 0
        self.STOP = 5
        
        self.MAX_LR = LA.norm(self.ub - self.lb, np.inf)
        
    def _grad(self):
        return self.gp.grad(self.w)
    
    def update(self, commit=False):
        assert self.STOP >= self.FAIL
        skip = 99
        clipnorm = 1
        
        implicit_path = []
        w = self.w

        for i in range(skip):
            gt = self.gp.grad_sample(w)
            if LA.norm(gt) > clipnorm:
                gt = gt*clipnorm / LA.norm(gt)
                
            if LA.norm(gt) < 1e-4*self.gp.dim:
                self.STOP = 0 # break out
                print("CAVEAT: negligible norm")
                
            lr = self.line(w, gt)
            print(lr)
            w_next = np.clip(w - lr*gt, self.lb, self.ub)
            
            post_f = self.gp.posterior(w)[0]
            post_f_next = self.gp.posterior(w_next)[0]
            
            # minimum posterior mean
            if post_f_next > post_f and len(implicit_path) >= 1:
                break
                
            w = w_next
            implicit_path.append(w)
        
        if commit == True:
            self.commit(w, implicit_path)
            
        return w, implicit_path
    
    def commit(self, w, path): # new query w, path arrive at w
        delta = np.abs(1e-2 * self.min)
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


    def line(self, w, gt, min_lr=0.0001, max_lr=1):
        max_lr = self.MAX_LR
        seed = np.random.randint(int(1e6))
        sobol = SobolEngine(1, scramble=True, seed=seed)

        n_mesh = 400
        LR = from_unit_cube(np.array(sobol.draw(n_mesh)), np.array([min_lr]), np.array([max_lr]))
        mesh = np.array([np.clip(w - lr*gt, self.lb, self.ub) for lr in LR])
        
        mu, covar = self.gp.posterior(mesh)
        mu = np.squeeze(mu)
        s = np.sqrt(np.diag(covar))
        L = np.linalg.cholesky(covar + 1e-6*np.eye(n_mesh)) # LL^T = Sigma (posterior covariance)
#         f_post = mu.reshape(-1,1) + np.dot(L, np.random.normal(size=(n_mesh, 1)))
        f_post = mu.reshape(-1, 1) 
        lr = LR[np.argmin(f_post)]
        return lr.item()


#     def line_ei(self, w, gt, min_lr=5e-3, max_lr=np.inf):
#         # find optimal lr via max PI
#         def subspace(lr):
#             this_best = min([self.f(w) for w in self.history_w])
#             return self.gp.EI(np.clip(w - lr*gt, self.lb, self.ub), this_best)
        
#         lr_cand = np.linspace(min_lr, max_lr, 30)
#         Score = np.array([subspace(lr) for lr in lr_cand])
#         lr = lr_cand[np.argmin(Score)]
        
#         Res = minimize(subspace, lr, bounds=[(min_lr, max_lr)], method="L-BFGS-B")
#         return Res.x.item()




#     def _pick_next_point(self, grad_, lr, y_best):
# #         assert (self.history_full_w[-1] == self.w).all()
#         clipnorm = 1
#         clipG = []
#         for g in grad_:
#             if LA.norm(g) > clipnorm:
#                 clipG.append(g*clipnorm/LA.norm(g))
#             else:
#                 clipG.append(g)
                
# #        1. vanilla GD
#         W = [np.clip(self.w - lr*g, self.lb, self.ub) for g in clipG]
        
# #        2.  heavy ball
# #         if len(self.history_w) == 1: 
# #             W = [np.clip(self.w - lr*g, self.lb, self.ub) for g in clipG]
# #         else:
# #             sgn = np.sign(self.f(self.history_w[-1]) - self.f(self.history_w[-2]))
# #             W = [np.clip(self.w - lr*g - sgn * (0.99/np.log(len(self.history_w) + np.e - 1)) * (self.w - self.history_full_w[-2]), self.lb, self.ub) for g in clipG]
        
# #         3. adam
# #         W, mt_, vt_ = self.adam(grad_, lr)  

# #         prev_true_grad = [optimize.approx_fprime(np.squeeze(w), lambda x: self.f(x).item(), eps) for w in optimizer.history_full_w]
# #         prev_grad_square = [self.gp.grad(w_prev)**2 for w_prev in self.history_full_w]
# #         Gt = functools.reduce(lambda x,y: 0.9*x + (1 - 0.9)*y, prev_grad_square)
# #         W = [np.clip(self.w - lr*g / np.sqrt(1e-8 + Gt), self.lb, self.ub) for g in grad_]
# #         print('lr = ', lr / np.sqrt(1e-8 + Gt))
        
            
#         Score = np.array([self.gp.PI(w, y_best) for w in W])
#         argmaxScore = np.argmax(Score)
# #         return W[argmaxScore], mt_[argmaxScore], vt_[argmaxScore], argmaxScore
#         return W[argmaxScore], argmaxScore
