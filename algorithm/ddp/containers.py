import sympy as sp
import numpy as np
from numba import jit, njit
from sympy import hessian, symbols

from .utils import *
from autograd import grad, jacobian


class j_h_m:

    def __init__(self, funcs):
        self.funcs = funcs

    # 获取雅克比矩阵
    def Ja(self,vars):
        jac_m = sp.zeros(len(vars), len(vars))
        jac_m = self.funcs.jacobian(vars)
        return jac_m

    # 获取海塞矩阵
    def His(self,vars1,vars2):
        His_m = sp.zeros(len(vars1), len(vars2))
        for i, fi in enumerate(self.funcs):
            for j, r in enumerate(vars2):
                for k, s in enumerate(vars1):
                    His_m[k, j] = sp.diff(sp.diff(fi, r), s)
        return His_m



class Dynamics:

    def __init__(self, f, f_x, f_u,f_xx,f_ux,f_uu):
        '''
           Dynamics container.
              f: Function approximating the dynamics.
              f_x: Partial derivative of 'f' with respect to state
              f_u: Partial derivative of 'f' with respect to action
              f_prime: returns f_x and f_u at once
        '''
        self.f = f
        self.f_x = f_x
        self.f_u = f_u
        self.f_xx = f_xx
        self.f_ux = f_ux
        self.f_uu = f_uu


        # f_x, f_u,f_xx,f_ux,f_uu = symbols('f_x,f_u,f_xx,f_ux,f_uu')
        self.f_prime1 = lambda x, u: (f_x(x, u), f_u(x, u))
        self.f_prime2 = lambda x, u: (f_xx(x, u), f_ux(x, u), f_uu(x, u))



    @staticmethod
    def SymDiscrete(f, x, u):
        '''
           Construct from Symbolic discrete time dynamics
        '''
        # 实例化类并计算雅克比矩阵
        h = j_h_m(f)
        f_x = h.Ja(x)
        f_u = h.Ja(u)
        f_xx = h.His(x,x)
        f_ux = h.His(u,x)
        f_uu = h.His(u,u)

        # f_ux = hessian(f,(u,x))


        # # 将sympy的符号转化为函数表达式
        # self.f_prime = sp.lambdify([f,f_x,f_u,f_xx,f_ux,f_uu],'numpy')  # 通过这段话转化为可以计算的函数表达式

        # # Partial derivatives of running cost
        # f_x = sp.jacobi(f,x)
        # f_u = sp.jacobi(f,u)
        # f_xx = sp.hessian(f,(x,x))
        # f_ux = sp.hessian(f,(u,x))
        # f_uu = sp.hessian(f,(u,u))
        #
        # f_x = f.jacobian(x)
        # f_u = f.jacobian(u)

        # f_xx = hessian(f,(x,x))
        # f_ux = hessian(f,(u,x))
        # f_uu = hessian(f,(u,u))

        # f_xx = f_x.jacobian(x)
        # f_uu = f_u.jacobian(u)
        # f_ux = f_u.jacobian(x)

        # f = sympy_to_numba(f, [x, u], True)
        # f_x = sympy_to_numba(f_x, [x, u], True)
        # f_u = sympy_to_numba(f_u, [x, u], True)
        # f_xx = sympy_to_numba(f_xx, [x, u],0)
        # f_ux = sympy_to_numba(f_ux, [x, u], 0)
        # f_uu = sympy_to_numba(f_uu, [x, u],0)


        # return Dynamics(f,f_x,f_u,f_xx,f_ux,f_uu)
        funs = [f, f_x, f_u, f_xx, f_ux, f_uu]
        for i in range(6):
            args = [x, u]
            redu = 0 if i in [3, 4, 5] else 1
            funs[i] = sympy_to_numba(funs[i], args, redu)
        return Dynamics(*funs)



    @staticmethod
    def SymContinuous(f, x, u, dt = 0.1):
        '''
           Construct from Symbolic continuous time dynamics
        '''
        # return Dynamics.SymDiscrete(x + f*dt, x, u)
        return Dynamics.SymDiscrete(f, x, u)

class Cost:

    def __init__(self, L, L_x, L_u, L_xx, L_ux, L_uu, Lf, Lf_x, Lf_xx):
        '''
           Container for Cost.
              L:  Running cost
              Lf: Terminal cost
        '''
        #Running cost and it's partial derivatives
        self.L = L
        self.L_x  = L_x
        self.L_u  = L_u
        self.L_xx = L_xx
        self.L_ux = L_ux
        self.L_uu = L_uu
        self.L_prime = lambda x, u: (L_x(x, u), L_u(x, u), L_xx(x, u), L_ux(x, u), L_uu(x, u))

        #Terminal cost and it's partial derivatives
        self.Lf = Lf
        self.Lf_x = Lf_x
        self.Lf_xx = Lf_xx
        self.Lf_prime = lambda x: (Lf_x(x), Lf_xx(x))


    @staticmethod
    def Symbolic(L, Lf, x, u):
        '''
           Construct Cost from Symbolic functions
        '''
        #convert costs to sympy matrices
        L_M  = sp.Matrix([L])
        Lf_M = sp.Matrix([Lf])

        #Partial derivatives of running cost
        L_x  = L_M.jacobian(x)
        L_u  = L_M.jacobian(u)
        L_xx = L_x.jacobian(x)
        L_ux = L_u.jacobian(x)
        L_uu = L_u.jacobian(u)

        #Partial derivatives of terminal cost
        Lf_x  = Lf_M.jacobian(x)
        Lf_xx = Lf_x.jacobian(x)

        #Convert all sympy objects to numba JIT functions
        funs = [L, L_x, L_u, L_xx, L_ux, L_uu, Lf, Lf_x, Lf_xx]
        for i in range(9):
          args = [x, u] if i < 6 else [x]
          redu = 0 if i in [3, 4, 5, 8] else 1
          funs[i] = sympy_to_numba(funs[i], args, redu)

        return Cost(*funs)

    @staticmethod
    def QR(Q, R, QT, x_goal, add_on = 0):
        '''
           Construct Quadratic cost
        '''
        x, u = GetSyms(Q.shape[0], R.shape[0])
        er = x - sp.Matrix(x_goal)
        L  = er.T@Q@er + u.T@R@u
        Lf = er.T@QT@er
        return Cost.Symbolic(L[0] + add_on, Lf[0], x, u)