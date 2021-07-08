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


# local search via GD
class GD:
    def __init__(self, gp, B, w, f):
        self.gp = gp
        self.B = B
        self.lb = B[:, 0]
        self.ub = B[:, 1]
        
        self.w = w # tracing GD from w
        self.f = f # query function
        
        self.history_w = [w]
        self.history_full_w = [w] # for momentum accelerated
        self.minSofar = f(w)
        self.Fail = 0
        self.stop = 4
        
        # for BFGS
#         self.H = np.eye(gp.dim)
#         self.B = np.eye(gp.dim)
        
        # times for contributions
        self.Contributions = 0
        
    def _grad(self):
        return self.gp.grad(self.w)
    
    def _clip(self):
        self.w = np.clip(self.w, self.lb, self.ub)

#     def adam(self, grad_, lr):
        '''
        Not finish 
        '''
#         t = len(self.history_full_w)
        
#         epsilon = 1e-8
#         b1 = 0.9
#         b2 = 0.999
        
#         prev_grad = [self.gp.grad(w_prev) for w_prev in self.history_full_w]
#         prev_grad_square = [self.gp.grad(w_prev)**2 for w_prev in self.history_full_w]
        
#         mt = functools.reduce(lambda x,y: b1*x + (1 - b1)*y, prev_grad)
#         vt = functools.reduce(lambda x,y: b2*x + (1 - b2)*y, prev_grad_square)
        
#         # bias corrected moment estimates
#         mhat = mt / (1 - b1 ** t) 
#         vhat = vt / (1 - b2 ** t)
                          
#         # update w
#         assert len(mhat_) == len(vhat_)
#         W = [np.clip(self.w - lr * mhat/np.sqrt(vhat+ epsilon), self.lb, self.ub) for i in range(len(mhat))]
#         return W, mt, vt
    
    
    def _pick_next_point(self, grad_, lr, y_best):
        assert (self.history_full_w[-1] == self.w).all()
        clipnorm = 1
        clip_grad_ = []
        for g in grad_:
            if LA.norm(g) > clipnorm:
                clip_grad_.append(g*clipnorm/LA.norm(g))
            else:
                clip_grad_.append(g)
                
#        1. vanilla GD
#         W = [np.clip(self.w - lr*g, self.lb, self.ub) for g in clip_grad_]
        
#        2.  heavy ball
        if len(self.history_w) == 1: 
            W = [np.clip(self.w - lr*g, self.lb, self.ub) for g in clip_grad_]
        else:
            sgn = np.sign(self.f(self.history_w[-1]) - self.f(self.history_w[-2]))
            W = [np.clip(self.w - lr*g - sgn * (0.99/np.log(len(self.history_w) + np.e - 1)) * (self.w - self.history_full_w[-2]), self.lb, self.ub) for g in clip_grad_]
        
        # 3. adam
#         W, mt_, vt_ = self.adam(grad_, lr)  

#         prev_true_grad = [optimize.approx_fprime(np.squeeze(w), lambda x: self.f(x).item(), eps) for w in optimizer.history_full_w]
#         prev_grad_square = [self.gp.grad(w_prev)**2 for w_prev in self.history_full_w]
#         Gt = functools.reduce(lambda x,y: 0.9*x + (1 - 0.9)*y, prev_grad_square)
#         W = [np.clip(self.w - lr*g / np.sqrt(1e-8 + Gt), self.lb, self.ub) for g in grad_]
#         print('lr = ', lr / np.sqrt(1e-8 + Gt))
        
        Score = np.array([self.gp.PI(w, y_best) for w in W])
        argmaxScore = np.argmax(Score)
#         return W[argmaxScore], mt_[argmaxScore], vt_[argmaxScore], argmaxScore
        return W[argmaxScore], argmaxScore

    
    def update_sample(self, y_best, lr=0.5):
        N = 1
        loop = 699
        delta = np.abs(1e-2 * self.minSofar)

        for i in range(loop):
            grad_ = [self.gp.grad_sample(self.w) for i in range(N)]
            self.w, argmaxScore = self._pick_next_point(grad_, lr, y_best) # GD
#             self.w, self.mt, self.vt, argmaxScore = self._pick_next_point(grad_, lr, y_best)
            self.history_full_w.append(self.w)
            m, v = self.gp.posterior_full_grad('full', self.gp.dim*[self.w])
#             if (np.squeeze(np.abs(m)) < 1*np.squeeze(np.diag(v))).any():
#                 break
            pr = np.mean(np.squeeze(np.abs(m)) - 2*np.squeeze(np.sqrt(np.diag(v))) > 0)
            if np.random.uniform(0, 1, 1).item() > pr:
                break
                
#         print('GD loop: ', i+1)
            
        self.history_w.append(self.w)
        
        # consecutive fail < 10
        if self.f(self.w) < self.minSofar - delta:
            self.Fail = 0
        else:
            self.Fail += 1
            
#         print('consecutive fail: ', self.Fail)
        
        self.minSofar = min(self.f(self.w), self.minSofar)
        
        # Max GD steps
        if len(self.history_full_w) > 49:
            self.stop = 1
            
        return self.w, -grad_[argmaxScore]
    
    
    def no_update_sample(self, y_best, lr=0.5):
        N = 1
        loop = 699
        delta = np.abs(1e-2 * self.minSofar)
        
        length = len(self.history_full_w)
        wTemp = self.w
     
        for i in range(loop):
            grad_ = [self.gp.grad_sample(self.w) for i in range(N)]
            self.w, argmaxScore = self._pick_next_point(grad_, lr, y_best) # GD
            self.history_full_w.append(self.w)
            m, v = self.gp.posterior_full_grad('full', self.gp.dim*[self.w])
            pr = np.mean(np.squeeze(np.abs(m)) - 2*np.squeeze(np.sqrt(np.diag(v))) > 0)
            if np.random.uniform(0, 1, 1).item() > pr:
                break  
        
        wRecord = self.w
        fullHistRecord = self.history_full_w.copy()
        
        # restore back
        self.history_full_w = self.history_full_w[:length]
        self.w = wTemp
        return wRecord, fullHistRecord

    
#     def update_bfgs(self, y_best, lr=0.1):
#         eps = np.sqrt(np.finfo(float).eps)
#         loop = 10
#         clipnorm = 1
#         self.H = np.eye(self.gp.dim)
#         self.B = np.eye(self.gp.dim)

#         for i in range(loop):
#             gt = self.gp.grad_sample(self.w)
#             if LA.norm(gt) > clipnorm:
#                 gt = gt*clipnorm/LA.norm(gt)
            
#             wt = self.w.reshape(self.gp.dim, 1)
#             wnew = np.clip(wt - lr*self.H@gt.reshape(self.gp.dim, 1), self.lb.reshape(self.gp.dim, 1), self.ub.reshape(self.gp.dim, 1))
            
#             # update H
#             st = wnew - wt
#             gnew = self.gp.grad_sample(np.squeeze(wnew)).reshape(self.gp.dim, 1)

#             if LA.norm(gnew) > clipnorm:
#                 gnew = gnew*clipnorm/LA.norm(gnew)

#             yt = gnew - gt.reshape(self.gp.dim, 1)
#             yt_ = optimize.approx_fprime(np.squeeze(wnew), lambda x: self.f(x).item(), eps).reshape(self.gp.dim, 1) - optimize.approx_fprime(np.squeeze(wt), lambda x: self.f(x).item(), eps).reshape(self.gp.dim, 1)

#             rho = 1/ (yt.T@st)
            
#             self.w = np.squeeze(wnew)
#             self.history_full_w.append(self.w)
#             m, v = self.gp.posterior_full_grad('full', self.gp.dim*[self.w])
#             if (np.abs(m) < 2.576*v).any():
#                 break
#             self.H = (np.eye(self.gp.dim) - rho*st@yt.T)@self.H@(np.eye(self.gp.dim) - rho*yt@st.T) + rho*st@st.T
#             self.B = (np.eye(self.gp.dim) - rho*st@yt_.T)@self.B@(np.eye(self.gp.dim) - rho*yt_@st.T) + rho*st@st.T
     
#         print('GD loop: ', i)
#         self.history_w.append(self.w)
        
#         delta = np.abs(1e-2 * self.minSofar)
#         if self.f(self.w) < self.minSofar - delta:
#             self.Fail = 0
#         else:
#             self.Fail += 1
        
#         self.minSofar = min(self.f(self.w), self.minSofar)
        
#         # Max GD steps
#         if len(self.history_full_w) > 200:
#             self.stop = 2
#         return self.w, - np.squeeze(gnew)
