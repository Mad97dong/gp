import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d.axes3d import Axes3D

from PIL import Image
from scipy.optimize import minimize
from GP import GP
from GP_grad import GP_grad
import time
import imageio

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
        self.contFail = 0
        
        # for adam
        self.mt = 0
        self.vt = 0
        
        # for BFGS
        self.H = np.eye(gp.dim)
        
    def _grad(self):
        return self.gp.grad(self.w)
    
    def _clip(self):
        self.w = np.clip(self.w, self.lb, self.ub)

    def adam(self, grad_, lr):
        t = len(self.history_full_w)
        
        epsilon = 1e-8
        b1 = 0.9
        b2 = 0.999

        mt_ = [b1 * self.mt + (1 - b1) * gt for gt in grad_]
        vt_ = [b2 * self.vt + (1 - b2) * np.square(gt) for gt in grad_]
        
        # bias corrected moment estimates
        mhat_ = [mt / (1 - b1 ** t) for mt in mt_]
        vhat_ = [vt / (1 - b2 ** t) for vt in vt_]
                          
        # update w
        assert len(mhat_) == len(vhat_)
        W = [np.clip(self.w - lr * mhat_[i]/(np.sqrt(vhat_[i]) + epsilon), self.lb, self.ub) for i in range(len(mhat_))]
        return W, mt_, vt_
    
    
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
        W = [np.clip(self.w - lr*g, self.lb, self.ub) for g in clip_grad_]
        
# #        2.  heavy ball
#         if len(self.history_w) == 1:
#             W = [np.clip(self.w - lr*g, self.lb, self.ub) for g in clip_grad_]
#         else:
#             sgn = np.sign(self.f(self.history_w[-1]) - self.f(self.history_w[-2]))
# #             print('heavy ball with lr = ', lr)
#             W = [np.clip(self.w - lr*g - sgn * 0.9*(self.w - self.history_full_w[-2]), self.lb, self.ub) for g in clip_grad_]
        
        # 3. adam
#         W, mt_, vt_ = self.adam(clip_grad_, lr)   

        
        Score = np.array([self.gp.PI(w, y_best) for w in W])
        argmaxScore = np.argmax(Score)
#         return W[argmaxScore], mt_[argmaxScore], vt_[argmaxScore], argmaxScore
        return W[argmaxScore], argmaxScore

    
    def update_sample(self, y_best, lr=0.5):
        N = 1
        loop = 7
        lr_decay = 1
#         if len(self.history_w) > 1:
#             if np.sign(self.f(self.history_w[-1]) - self.f(self.history_w[-2])) > 0:
#                 lr_decay = 0.95
#                 lr = lr / 2

        for i in range(loop):
            grad_ = [self.gp.grad_sample(self.w) for i in range(N)]
#             lr = lr * lr_decay
#             self.w, argmaxScore = self._pick_next_point(grad_, lr, y_best) # GD
            self.w, argmaxScore = self._pick_next_bfgs(grad_, lr, y_best) # BFGS
#             self.w, self.mt, self.vt, argmaxScore = self._pick_next_point(grad_, lr, y_best)
            self.history_full_w.append(self.w)
            m, v = self.gp.posterior_full_grad('full', self.gp.dim*[self.w])
            if (np.abs(m) < 1*v).any():
                break
     
        print('GD loop: ', i)
        self.history_w.append(self.w)
        
        delta = 1e-3 * self.minSofar
        if self.f(self.w) < self.minSofar - delta:
            self.contFail = 0
        else:
            self.contFail += 1
        
        self.minSofar = min(self.f(self.w), self.minSofar)
        return self.w, -grad_[argmaxScore]
    
    
    def _pick_next_bfgs(self, grad_, lr, y_best):
        assert (self.history_full_w[-1] == self.w).all()
        clipnorm = 1
        clip_grad_ = []
        for g in grad_:
            if LA.norm(g) > clipnorm:
                clip_grad_.append(g*clipnorm/LA.norm(g))
            else:
                clip_grad_.append(g)
                
        W = [np.clip(self.w.T - lr*self.H@g.T, self.lb, self.ub) for g in clip_grad_]
        Score = np.array([self.gp.PI(w, y_best) for w in W])
        argmaxScore = np.argmax(Score)
        
        # update H
        w = W[argmaxScore]
        st = w - self.w.T
        
        gnew = self.gp.grad_sample(w).T
        if LA.norm(gnew) > clipnorm:
            gnew = gnew*clipnorm/LA.norm(g)

        yt = gnew - clip_grad_[argmaxScore].T
        
        rho = 1/ (np.dot(yt, st))
        V = np.eye(gp.self.dim) - rho*st@yt.T
        self.H = V@self.H@V.T + rho*st@st.T
        return W[argmaxScore], argmaxScore
    
