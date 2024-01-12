'''
 swing up pendulum with limited torque
'''
import sympy as sp
import numpy as np

from ilqr_j import iLQR
from ilqr_j.utils import GetSyms, Constrain, Bounded
from ilqr_j.containers import Dynamics, Cost

#state and action dimensions
n_x = 3
n_u = 1
#time step
dt = 0.025

#Construct pendulum dynamics
m, g, l = 1, 10, 1
def f(x, u):
    #current state
    sin, cos, omega = x
    theta = np.arctan2(sin, cos)
    #angular acceleration
    alpha = (u[0] - m*g*l*np.sin(theta + np.pi))/(m*l**2)
    #next theta
    theta_n = theta + omega*dt
    #return next state
    return np.array([np.sin(theta_n),
                     np.cos(theta_n),
                     omega + alpha*dt])
#call dynamics container
Pendulum = Dynamics.Discrete(f)


#Construct cost to swing up Pendulum
x, u = GetSyms(n_x, n_u)
#theta = 0 --> sin(theta) = 0, cos(theta) = 1
x_goal = np.array([0, 1, 0])
Q  = np.diag([0, 1, 0.1])
R  = np.diag([0.1])
QT = np.diag([0, 100, 100])
#Add constraints on torque input (2Nm to -2Nm)
cons = Bounded(u, high = [2], low = [-2])
SwingUpCost = Cost.QR(Q, R, QT, x_goal, cons)


#initialise the controller
controller = iLQR(Pendulum, SwingUpCost)

#initial state
#theta = pi --> sin(theta) = 0, cos(theta) = -1
x0 = np.array([0, -1, 0])
#initial guess
us_init = np.random.randn(200, n_u)*0.01
#get optimal states and actions
xs, us, cost_trace = controller.fit(x0, us_init)


#Plot theta and action trajectory
import matplotlib.pyplot as plt
# theta = np.arctan2(xs[:, 0], xs[:, 1])
# theta = np.where(theta < 0, 2*np.pi+theta, theta)
# plt.plot(theta)
# plt.plot(us)
# plt.show()
plt.figure(0)
plt.plot(xs[:,0],xs[:,1],label='LQR px-py',linewidth=2.5)
# plt.plot(xr[:,0],xr[:,1],label='ref px-py',linewidth=2.5)
plt.legend(fontsize='large', loc='upper right', prop={'family': 'Times New Roman', 'size': 25})
plt.xlabel("px [m]", fontdict={'family': 'Times New Roman', 'size': 20})
plt.ylabel("py [m]", fontdict={'family': 'Times New Roman', 'size': 20})

plt.figure(1)
plt.plot(xs[:,0],label='LQR px',linewidth=2.5)
# plt.plot(xr[:,0],label='ref px',linewidth=2.5)
plt.legend(fontsize='large', loc='upper right', prop={'family': 'Times New Roman', 'size': 25})
plt.xlabel("time [s]", fontdict={'family': 'Times New Roman', 'size': 20})
plt.ylabel("px [m]", fontdict={'family': 'Times New Roman', 'size': 20})

plt.figure(2)
plt.plot(xs[:,1],label='LQR py',linewidth=2.5)
# plt.plot(xr[:,1],label='ref py',linewidth=2.5)
plt.legend(fontsize='large', loc='upper right', prop={'family': 'Times New Roman', 'size': 25})
plt.xlabel("time [s]", fontdict={'family': 'Times New Roman', 'size': 20})
plt.ylabel("py [m]", fontdict={'family': 'Times New Roman', 'size': 20})

plt.figure(3)
plt.plot(xs[:,2],label='LQR heading',linewidth=2.5)
# plt.plot(xr[:,2],label='ref heading',linewidth=2.5)
plt.legend(fontsize='large', loc='upper right', prop={'family': 'Times New Roman', 'size': 25})
plt.xlabel("time [s]", fontdict={'family': 'Times New Roman', 'size': 20})
plt.ylabel("heading [rad]", fontdict={'family': 'Times New Roman', 'size': 20})
plt.show()