import numba
import numpy as np
from numba import jit
import time

class DDP:

    def __init__(self, dynamics, cost):
        '''
           iterative Linear Quadratic Regulator
           Args:
             dynamics: dynamics container
             cost: cost container
        '''
        self.cost = cost
        self.dynamics = dynamics
        self.params = {'alphas'  : 0.5**np.arange(8), #line search candidates
                       'regu_init': 20,    #initial regularization factor
                       'max_regu' : 10000,
                       'min_regu' : 0.001}

    def fit(self, x0, us_init, maxiters = 100, early_stop = True):
        '''
        Args:
          x0: initial state
          us_init: initial guess for control input trajectory
          maxiter: maximum number of iterations
          early_stop: stop early if improvement in cost is low.

        Returns:
          xs: optimal states
          us: optimal control inputs
          cost_trace: cost trace of the iterations
        '''
        return run_ddp(self.dynamics.f, self.dynamics.f_prime1, self.dynamics.f_prime2,
                       self.cost.L,self.cost.Lf, self.cost.L_prime, self.cost.Lf_prime,
                        x0, us_init, maxiters, early_stop, **self.params)

    def rollout(self, x0, us):
        '''
        Args:
          x0: initial state
          us: control input trajectory

        Returns:
          xs: rolled out states
          cost: cost of trajectory
        '''
        return rollout(self.dynamics.f, self.cost.L, self.cost.Lf, x0, us)

# @jit
def run_ddp(f, f_prime1,f_prime2, L, Lf, L_prime, Lf_prime, x0, u_init, max_iters, early_stop,
             alphas, regu_init = 20, max_regu = 10000, min_regu = 0.001):
    '''
       iLQR main loop
    '''
    us = u_init
    regu = regu_init
    xr = [x0]
    # First forward rollout
    xs, J_old = rollout(f, L, Lf, x0, us)
    # cost trace
    cost_trace = [J_old]
    # Run main loop
    for it in range(max_iters):
        dt = time.time()
        ks, Ks, exp_cost_redu = backward_pass(f,f_prime1,f_prime2, L_prime, Lf_prime, xs, us, regu)

        # Early termination if improvement is small
        if it > 3 and early_stop and np.abs(exp_cost_redu) < 1e-5: break

        # Backtracking line search
        for alpha in alphas:
          xs_new, us_new, J_new = forward_pass(f, L, Lf, xs, us, ks, Ks, alpha)
          if J_old - J_new > 0:
              # Accept new trajectories and lower regularization
              J_old = J_new
              xs = xs_new
              us = us_new
              regu *= 0.7
              break
        else:
            # Reject new trajectories and increase regularization
            regu *= 2.0

        cost_trace.append(J_old)
        regu = min(max(regu, min_regu), max_regu)
        tt = (time.time() - dt) * 1000
        print("iteration step:", it + 1)
        print("time of every step:", tt, "ms")

    return xs, us, cost_trace


# @jit
def rollout(f, L, Lf, x0, us):
    '''
      Rollout with initial state and control trajectory
    '''
    xs = np.empty((us.shape[0] + 1, x0.shape[0]))
    xs[0] = x0
    cost = 0
    for n in range(us.shape[0]):
      xs[n+1] = f(xs[n], us[n])
      cost += L(xs[n], us[n])
    cost += Lf(xs[-1])
    return xs, cost


# @numba.njit
# @jit
def forward_pass(f, L, Lf, xs, us, ks, Ks, alpha):
    '''
       Forward Pass
    '''
    xs_new = np.empty(xs.shape)

    cost_new = 0.0
    xs_new[0] = xs[0]
    us_new = us + alpha*ks

    for n in range(us.shape[0]):
        us_new[n] += Ks[n].dot(xs_new[n] - xs[n])
        xs_new[n + 1] = f(xs_new[n], us_new[n])
        cost_new += L(xs_new[n], us_new[n])

    cost_new += Lf(xs_new[-1])

    return xs_new, us_new, cost_new


# @jit
def backward_pass(f, f_prime1,f_prime2, L_prime, Lf_prime, xs, us, regu):
    '''
       Backward Pass
    '''
    ks = np.empty(us.shape)
    Ks = np.empty((us.shape[0], us.shape[1], xs.shape[1]))

    delta_V = 0
    V_x, V_xx = Lf_prime(xs[-1])
    regu_I = regu*np.eye(V_xx.shape[0])
    for n in range(us.shape[0] - 1, -1, -1):
        f_x, f_u= f_prime1(xs[n], us[n])
        f_xx, f_ux, f_uu = f_prime2(xs[n], us[n])
        l_x, l_u, l_xx, l_ux, l_uu  = L_prime(xs[n], us[n])

        # Q_terms
        Q_x  = l_x  + f_x.T@V_x
        Q_u  = l_u  + f_u.T@V_x

        # V=np.insert(V_x, 0, V_x)
        # V=np.column_stack((V_x, V_x))
        # Q_xx = l_xx + f_x.T @ V_xx @ f_x + np.tensordot(V_x.T, f_xx, axes=1)
        # V_x= V_x.reshape(V_x.shape[0], 1)
        # Q_ux = l_ux + f_u.T @ V_xx @ f_x + np.tensordot(V_x, f_ux,  axes=-1)
        # Q_uu = l_uu + f_u.T @ V_xx @ f_u + np.tensordot(V_x,f_uu, axes=-1)
        Q_xx = l_xx + f_x.T@V_xx@f_x+np.dot(V_x,np.squeeze(f_xx))
        Q_ux = l_ux + f_u.T @ V_xx @ f_x + np.dot(np.squeeze(f_ux), V_x.reshape(V_x.shape[0], 1))
        # V_x = V_x.reshape(2,5)
        Q_uu = l_uu + f_u.T @ V_xx @ f_u
        # Q_uu = l_uu + f_u.T@V_xx@f_u+np.dot(np.squeeze(f_uu),V_x)



        # gains
        f_u_dot_regu = f_u.T@regu_I
        Q_ux_regu = Q_ux + f_u_dot_regu@f_x
        Q_uu_regu = Q_uu + f_u_dot_regu@f_u
        Q_uu_inv = np.linalg.inv(Q_uu_regu)


        k = -Q_uu_inv @ Q_u
        K = -Q_uu_inv @ Q_ux_regu
        # ks[n] = ks[n].reshape(ks[n].shape[0], 1)
        # Ks[n] = Ks[n].reshape(Ks[n].shape[0], 1)
        ks[n], Ks[n] = k, K

        # V_terms
        V_x  = Q_x + K.T@Q_u + Q_ux.T@k + K.T@Q_uu@k
        V_xx = Q_xx + 2*K.T@Q_ux + K.T@Q_uu@K
        #expected cost reduction
        delta_V += Q_u.T@k + 0.5*k.T@Q_uu@k
        # np.seterr(invalid='ignore')
    return ks, Ks, delta_V
