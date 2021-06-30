import numpy as np
from numpy import *
import math
from numpy.matlib import *
from scipy.stats import multivariate_normal


# 2d
class sincos:
    def __init__(self, noise=False, noise_std=0):
        self.input_dim = 2
        self.bounds = {"x": (2, 10), "y": (2, 10)}
        self.name = "sincos"
        self.noise = noise
        self.noise_std = noise_std

    def func(self, coord):
        if coord.ndim == 1:
            coord = coord[np.newaxis, :]
        X1 = coord[:, 0]
        X2 = coord[:, 1]
        n = coord.shape[0]
        #  std = 0.05*self.findSdev()
        noise_val = 0
        if self.noise == True:
            noise_val = np.random.normal(0, self.noise_std, n)
        else:
            noise_val = 0
        
#         print('X1', X1)
#         print('X2', X2)
#         out = ((np.sin(X1) * np.cos(X2)) / np.sqrt(X1 * X2)) + noise_val
        out = np.sin(X1) * np.cos(X2) + noise_val
        return out
    
class quad:
    def __init__(self, noise=False, noise_std=0):
        self.input_dim = 2
        self.bounds = {"x": (-30, 30), "y": (-30, 30)}
        self.name = "quad"
        self.noise = noise
        self.noise_std = noise_std

    def func(self, coord):
        if coord.ndim == 1:
            coord = coord[np.newaxis, :]
            
        X1 = coord[:, 0]
        X2 = coord[:, 1]
        n = coord.shape[0]
        #  std = 0.05*self.findSdev()
        noise_val = 0
        if self.noise == True:
            noise_val = np.random.normal(0, self.noise_std, n)
        else:
            noise_val = 0

        return 0.5 * X1**2 + 4.5 * X2**2 + noise_val
    
class Keane:
    def __init__(self, noise=False, noise_std=0):
        self.input_dim = 2
        self.bounds = {"x": (1e-6, 10), "y": (1e-6, 10)}
        self.name = "Keane"
        self.noise = noise
        self.noise_std = noise_std

    def func(self, coord):
        if coord.ndim == 1:
            coord = coord[np.newaxis, :]
        x = coord[:, 0]
        y = coord[:, 1]
        out = np.sin(x-y)**2 * np.sin(x+y)**2 / np.sqrt(x**2 + y**2)
        
        n = coord.shape[0]
        noise_val = 0
        if self.noise == True:
            noise_val = np.random.normal(0, self.noise_std, n)
        else:
            noise_val = 0
            
        return -out + noise_val

class Rosenbrock_2:
    def __init__(self, noise=False, noise_std=0):
        self.input_dim = 2
        self.bounds = {'x': (-5, 10), 'y':  (-5, 10)}
        self.name = 'Rosenbrock'
        self.noise = noise
        self.noise_std = noise_std
        
    def func(self,coord):
        if coord.ndim == 1:
            coord = coord[np.newaxis, :]
        x = coord[:, 0]
        y = coord[:, 1]
        
        n = coord.shape[0]
        noise_val = 0
        if self.noise == True:
            noise_val = np.random.normal(0, self.noise_std, n)
        else:
            noise_val = 0
            
        return 100.0*(y-x**2.0)**2.0 + (1-x)**2.0 + noise_val
    
    
class Shubert_2:
    def __init__(self, noise=False, noise_std=0):
        self.input_dim = 2
        
        self.bounds = {"x": (-5.12, 5.12), "y": (-5.12, 5.12)}
        self.bounds = {"x": (-1, 2), "y": (-1, 2)}
        self.name = "Shubert"
        self.noise = noise
        self.noise_std = noise_std
        
    def func(self, coord):
        if coord.ndim == 1:
            coord = coord[np.newaxis, :]
                        
        X1 = coord[:, 0]
        X2 = coord[:, 1]
        n = coord.shape[0]
        
        out_0 = 0
        out_1 = 0
        for i in range(1, 6):
            out_0 += i * np.cos((i + 1) * X1 + i)
        for i in range(1, 6):
            out_1 += i * np.cos((i + 1) * X2 + i)
        return out_0 * out_1


    
    
    
class Branin:
    def __init__(self, noise=False, noise_std=0):
        self.input_dim = 2
        self.bounds = {"x": (-5, 10), "y": (0, 15)}
        self.name = "Branin"
        self.noise = noise
        self.noise_std = noise_std

    def func(self, coord):
        if coord.ndim == 1:
            coord = coord[np.newaxis, :]
                        
        X1 = coord[:, 0]
        X2 = coord[:, 1]
        n = coord.shape[0]
        
        PI = math.pi
        a = 1
        b = 5.1 / (4 * pow(PI, 2))
        c = 5 / PI
        r = 6
        s = 10
        t = 1 / (8 * PI)
        out = (
            a * (X2 - b * X1 ** 2 + c * X1 - r) ** 2
            + s * (1 - t) * np.cos(X1)
            + s
        )
        noise_val = 0
        if self.noise == True:
            noise_val = np.random.normal(0, self.noise_std, n)
        else:
            noise_val = 0
        return out + noise_val

    
class Ackley_2:
    def __init__(self, noise=False, noise_std=0):
        self.input_dim = 2
#         self.bounds = {"x": (-32.768, 32.768), "y": (-32.768, 32.768)}
        self.bounds = {"x": (-4, 4), "y": (-4, 4)} 
        self.name = "Ackley_2d"
        self.noise = noise
        self.noise_std = noise_std

    def func(self, coord):
        x = np.asarray(coord).reshape((-1, self.input_dim))
        
        firstMean = np.array([np.mean(k**2) for k in x]).reshape(-1, 1)
        secondMean = np.array([np.mean(np.cos(2*np.pi*k)) for k in x]).reshape(-1, 1)
  
        fx = np.reshape(-20 * np.exp(-0.2*np.sqrt(firstMean)) - np.exp(secondMean) + 20 + np.e, (-1, 1))
        if self.noise:
            return (fx + np.random.normal(0, self.noise_std, size=(x.shape[0], 1)))
        else:
            return fx
       

class Eggholder:
    def __init__(self, noise=False, noise_std=0):
        self.input_dim = 2
        self.bounds = {"x": (-5, 5), "y": (-5, 5)}
        self.name = "Eggholder"
        self.noise = noise
        self.noise_std = noise_std
        
    def func(self, coord):
        if coord.ndim == 1:
            coord = coord[np.newaxis, :]
                        
        X1 = coord[:, 0]
        X2 = coord[:, 1]
        n = coord.shape[0]
        if self.noise == True:
            noise_val = np.random.normal(0, self.noise_std, n)
        else:
            noise_val = 0
        return X1**2 + X2**2 + 25* np.sin(X1)**2 + 25* np.sin(X2)**2 + noise_val



class Dropwave:
    def __init__(self):
        self.input_dim = 2
        self.bounds = {"x": (-5.12, 5.12), "y": (-5.12, 5.12)}
        self.name = "Dropwave"

    def func(self, coord):
        if len(coord.shape) == 1:
            x1 = coord[0]
            x2 = coord[1]
        else:
            x1 = coord[:, 0]
            x2 = coord[:, 1]

        fval = -(1 + np.cos(12 * np.sqrt(x1 ** 2 + x2 ** 2))) / (
            0.5 * (x1 ** 2 + x2 ** 2) + 2
        )
        return -fval


class Shekel:
    def __init__(self):
        self.input_dim = 4
        self.bounds = {"x": (0, 10), "y": (0, 10), "z": (0, 10), "a": (0, 10)}
        self.name = "Shekel"

    def func(self, coord):
        m = 5
        C = [[4, 1, 8, 6, 3], [4, 1, 8, 6, 7], [4, 1, 8, 6, 3], [4, 1, 8, 6, 7]]
        B = [0.1, 0.2, 0.2, 0.4, 0.4]
        outer = 0
        for i in range(m):
            inner = 0
            for j in range(self.input_dim):
                inner = inner + (coord[j] - C[j][i]) ** 2
            outer = outer + (1 / (inner + B[i]))

        return outer

class cos:
    def __init__(self):
        self.input_dim = 1
        self.bounds = {"x": (-1, 15)}
        # self.bounds={'x':(0,1)}
        self.name = "sin"

    def func(self, coord):
        x = np.asarray(coord)
        fval = np.cos(x)
        return fval


class Ackley_1:
    def __init__(self, noise=False, noise_std=0):
        self.input_dim = 1
        self.bounds = {"x": (-4, 4)}
        self.name = "Ackley_1d"
        self.noise = noise
        self.noise_std = noise_std

    def func(self, coord):
        x = np.asarray(coord).reshape((-1, 1))
        fx = np.reshape(-20 * np.exp(-0.2*np.sqrt(x**2)) - np.exp(np.cos(2*np.pi*x)) + 20 + np.e, (-1, 1))
        if self.noise: 
            return (fx + np.random.normal(0, self.noise_std, size=(x.shape[0], 1)) ) / 10
        else:
            return fx / 10

    
class sin:
    def __init__(self, noise=False, noise_std=0):
        np.random.seed(0)
        self.input_dim = 1
        self.bounds = {"x": (-1, 15)}
        # self.bounds={'x':(0,1)}
        self.name = "sin"
        self.noise = noise
        self.noise_std = noise_std

    def func(self, coord):
        x = np.asarray(coord)
        fval = np.reshape(np.sin(x), (-1, 1))
        if self.noise:
            return fval + np.random.normal(0, self.noise_std, size=(x.shape[0], 1))
        else:
            return fval

class sin_add:
    def __init__(self, noise=False, noise_std=0):
        np.random.seed(0)
        self.input_dim = 1
        self.bounds = {"x": (-1, 15)}
        # self.bounds={'x':(0,1)}
        self.name = "sin_add"
        self.noise = noise
        self.noise_std = noise_std

    def func(self, coord):
        x = np.asarray(coord)
        fval = np.reshape(np.sin(x), (-1, 1)) + 0.1 * np.reshape(x, (-1, 1))
        if self.noise:
            return fval + np.random.normal(0, self.noise_std, size=(x.shape[0], 1))
        else:
            return fval



    
    
# High Dimensional
class Hartmann_3:
    def __init__(self):
        self.input_dim = 3
        self.bounds = {"x": (0, 1), "y": (0, 1), "z": (0, 1)}
        self.name = "Hartmann_3"

    def func(self, coord):
        c = array([1, 1.2, 3, 3.2])
        A = array([[3, 10, 30], [0.1, 10, 35], [3, 10, 30], [0.1, 10, 35]])
        P = array(
            [
                [0.3689, 0.1170, 0.2673],
                [0.4699, 0.4387, 0.747],
                [0.1091, 0.8732, 0.5547],
                [0.0382, 0.5743, 0.8828],
            ]
        )
        out = sum(c * exp(-sum(A * (repmat(coord, 4, 1) - P) ** 2, axis=1)))
        return out


class Hartmann_6:
    def __init__(self, noise=False, noise_std=0):
        self.input_dim = 6
        self.bounds = {
            "x": (0, 1),
            "y": (0, 1),
            "z": (0, 1),
            "a": (0, 1),
            "b": (0, 1),
            "c": (0, 1),
        }
        self.name = "Hartmann_6"
        self.noise = noise
        self.noise_std = noise_std

    def func(self, coord):
        coord = np.array(coord)
        xx = coord
        if len(xx.shape) == 1:
            xx = xx.reshape((1, 6))

        assert xx.shape[1] == 6

        n = xx.shape[0]
        y = np.zeros(n)
        for i in range(n):
            alpha = np.array([1.0, 1.2, 3.0, 3.2])
            A = np.array(
                [
                    [10, 3, 17, 3.5, 1.7, 8],
                    [0.05, 10, 17, 0.1, 8, 14],
                    [3, 3.5, 1.7, 10, 17, 8],
                    [17, 8, 0.05, 10, 0.1, 14],
                ]
            )
            P = 1e-4 * np.array(
                [
                    [1312, 1696, 5569, 124, 8283, 5886],
                    [2329, 4135, 8307, 3736, 1004, 9991],
                    [2348, 1451, 3522, 2883, 3047, 6650],
                    [4047, 8828, 8732, 5743, 1091, 381],
                ]
            )

            outer = 0
            for ii in range(4):
                inner = 0
                for jj in range(6):
                    xj = xx[i, jj]
                    Aij = A[ii, jj]
                    Pij = P[ii, jj]
                    inner = inner + Aij * (xj - Pij) ** 2

                new = alpha[ii] * np.exp(-inner)
                outer = outer + new

            y[i] = -outer
            
        noise_val = 0
        if self.noise == True:
            noise_val = np.random.normal(0, self.noise_std, xx.shape[0])
        else:
            noise_val = 0
        return y + noise_val


class Alpine:
    def __init__(self, noise=False, noise_std=0):
        self.input_dim = 20
        self.bounds = dict(zip([str(i) for i in range(self.input_dim)], self.input_dim*[(0, 10)]))
        self.name = "Alpine"
        self.noise = noise
        self.noise_std = noise_std

    def func(self, coord):
        if coord.ndim == 1:
            coord = coord.reshape(1, -1)
        fitness = np.zeros(coord.shape[0])
        for i in range(self.input_dim):
            fitness += np.abs(0.1 * coord[:, i] + coord[:, i] * np.sin(coord[:, i]))
            
        noise_val = 0
        if self.noise == True:
            noise_val = np.random.normal(0, self.noise_std, coord.shape[0])
        else:
            noise_val = 0
        return fitness + noise_val


class Ackley:
    def __init__(self, noise=False, noise_std=0):
        self.input_dim = 20
        self.bounds = dict(zip([str(i) for i in range(self.input_dim)], self.input_dim*[(-5, 5)])) # -32.768, 32.768
        self.name = "Ackley_6"
        self.noise = noise
        self.noise_std = noise_std

    def func(self, coord):
        if coord.ndim == 1:
            coord = coord.reshape(1, -1)
    
        firstSum = 0.0
        secondSum = 0.0
        
        dim = coord.shape[1]
        n = float(len(coord))
        
        for i in range(dim):
            firstSum += coord[:, i] ** 2.0
            secondSum += np.cos(2.0 * np.pi * coord[:, i])
        
        out = - 20.0 * np.exp(-0.2 * np.sqrt(firstSum / dim)) - np.exp(secondSum / dim) + 20 + np.exp(1)
    
        if self.noise == True:
            noise_val = np.random.normal(0, self.noise_std, coord.shape[0])
        else:
            noise_val = 0
        return out + noise_val

    
class Michalewicz:
    def __init__(self, noise=False, noise_std=0):
        self.input_dim = 2
        self.bounds = dict(zip([str(i) for i in range(self.input_dim)], self.input_dim*[(0, np.pi)]))
        self.name = "Michalewicz"
        self.noise = noise
        self.noise_std = noise_std

    def func(self, coord):
        if coord.ndim == 1:
            coord = coord.reshape(1, -1)
                   
        if self.noise == True:
            noise_val = np.random.normal(0, self.noise_std, coord.shape[0])
        else:
            noise_val = 0
        
        out = 0
        for j in range(self.input_dim):
            out += np.sin(coord[:, j]) * np.sin(j * coord[:, j] ** 2 / np.pi) ** (2 * 10)
        
        if self.noise == True:
            noise_val = np.random.normal(0, self.noise_std, coord.shape[0])
        else:
            noise_val = 0
        return -out + noise_val


    
class Schwefel:
    def __init__(self, noise=False, noise_std=0, dim=2):
        self.input_dim = dim
        self.bounds = dict(zip([str(i) for i in range(self.input_dim)], self.input_dim*[(400, 500)]))
        self.name = "Schwefel"
        self.noise = noise
        self.noise_std = noise_std


    def func(self, coord):
        alpha = 418.982887
        if coord.ndim == 1:
            coord = coord.reshape(1, -1)
            
        if self.noise == True:
            noise_val = np.random.normal(0, self.noise_std, coord.shape[0])
        else:
            noise_val = 0
        
        out = np.ones(coord.shape[0]) * alpha * self.input_dim
        for i in range(coord.shape[0]): # each row
            out[i] -= np.sum( coord[i]*np.sin(np.sqrt(np.abs(coord[i]))) )
        
        if self.noise == True:
            noise_val = np.random.normal(0, self.noise_std, coord.shape[0])
        else:
            noise_val = 0
        return out + noise_val
    
    
class Rosenbrock:
    def __init__(self, noise=False, noise_std=0):
        self.input_dim = 100
        self.bounds = dict(zip([str(i) for i in range(self.input_dim)], self.input_dim*[(-5, 10)]))
        self.name = 'Rosenbrock'
        self.noise = noise
        self.noise_std = noise_std
        
    def func(self,coord):
        if coord.ndim == 1:
            coord = coord.reshape(1, -1)
        
        out = 0
        for i in range(0, self.input_dim - 2):
            
            x = coord[:, i]
            y = coord[:, i+1]
            out += 100.0*(y-x**2.0)**2.0 + (1-x)**2.0
        
        if self.noise == True:
            noise_val = np.random.normal(0, self.noise_std, coord.shape[0])
        else:
            noise_val = 0
            
        return out + noise_val
    
    
class Griewank:
    def __init__(self, noise=False, noise_std=0):
        self.input_dim = 20
        self.bounds = dict(zip([str(i) for i in range(self.input_dim)], self.input_dim*[(-10, 10)]))
        self.name = 'Griewank'
        self.noise = noise
        self.noise_std = noise_std
        
    def func(self, coord):
        if coord.ndim == 1:
            coord = coord.reshape(1, -1)
        
        out1 = 1
        for i in range(0, self.input_dim):
            out1 += coord[:, i]**2/4000
        
        out2 = 1
        for i in range(0, self.input_dim):
            out2 *= np.cos(coord[:, i]/np.sqrt(i+1))
        
        if self.noise == True:
            noise_val = np.random.normal(0, self.noise_std, coord.shape[0])
        else:
            noise_val = 0
            
        return out1 - out2 + noise_val
    
    
class Levy:
    def __init__(self, noise=False, noise_std=0, dim=2):
        self.input_dim = dim
        self.bounds = dict(zip([str(i) for i in range(self.input_dim)], self.input_dim*[(-5, 10)]))
        self.lb = -5 * np.ones(dim)
        self.ub = 10 * np.ones(dim)
        self.name = 'Levy'
        self.noise = noise
        self.noise_std = noise_std
        
    def func(self, coord):
        eps = np.sqrt(np.finfo(float).eps)
        if coord.ndim == 1:
            coord = coord.reshape(1, -1)
            assert np.all(coord >= self.lb - eps) and np.all(coord <= self.ub + eps)
        else:
            for i in range(coord.shape[0]):
                assert np.all(coord[i] >= self.lb - eps) and np.all(coord[i] <= self.ub + eps)
        
        if self.noise == True:
            noise_val = np.random.normal(0, self.noise_std, coord.shape[0])
        else:
            noise_val = 0
            
        w = 1 + (coord - 1.0) / 4.0
        val = np.sin(np.pi * w[:, 0]) ** 2 + (w[:, self.input_dim - 1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[:, self.input_dim - 1])**2)
        for i in range(1, self.input_dim-1):
            val += (w[:, i] - 1)**2 * (1 + 10* np.sin(np.pi * w[:, i] + 1)**2)
        return val + noise_val
    
    
    
class Rastrigin:
    def __init__(self, noise=False, noise_std=0, dim=2):
        self.input_dim = dim
        self.bounds = dict(zip([str(i) for i in range(self.input_dim)], self.input_dim*[(-5.12, 5.12)]))
        self.lb = -5.12 * np.ones(dim)
        self.ub = 5.12 * np.ones(dim)
        self.name = 'Rastrigin'
        self.noise = noise
        self.noise_std = noise_std
        
    def func(self, coord):
        eps = np.sqrt(np.finfo(float).eps)
        if coord.ndim == 1:
            coord = coord.reshape(1, -1)
            assert np.all(coord >= self.lb - eps) and np.all(coord <= self.ub + eps)
        else:
            for i in range(coord.shape[0]):
                assert np.all(coord[i] >= self.lb - eps) and np.all(coord[i] <= self.ub + eps)
        
        if self.noise == True:
            noise_val = np.random.normal(0, self.noise_std, coord.shape[0])
        else:
            noise_val = 0
        
        val = 10*self.input_dim
        for i in range(0, self.input_dim):
            val += coord[:, i]**2 - 10*np.cos(2*np.pi*coord[:, i])
        return val + noise_val

# class Shubert:
#     def __init__(self, noise=False, noise_std=0):
#         self.input_dim = 10
#         self.bounds = dict(zip([str(i) for i in range(self.input_dim)], self.input_dim*[(-5.12, 5.12)]))
#         self.name = "Shubert"
#         self.noise = noise
#         self.noise_std = noise_std
        
#     def func(self, coord):
#         if coord.ndim == 1:
#             coord = coord.reshape(1, -1)
        
#         def f(x):
#             out = 0
#             for j in range(1, 6):
#                 out += j * np.cos((j + 1) * x + j)
#             return out
            
#         out = np.ones(coord.shape[0])
#         for i in range(self.input_dim):
#             out = out * f(coord[:, i])   
            
#         if self.noise == True:
#             noise_val = np.random.normal(0, self.noise_std, coord.shape[0])
#         else:
#             noise_val = 0
#         return out + noise_val