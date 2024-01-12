'''
 Vehicle Overtaking
 Adjust cost and initial state to get desired behaviors
'''

import sympy as sp
import numpy as np
from ilqr_j import *
import matplotlib.pyplot as plt



def vehicle_kinematics(state, action):
    px, py, heading, vel, steer = state
    accel, steer_vel = action

    state_dot = sp.Matrix([
                    vel*sp.cos(heading),
                    vel*sp.sin(heading),
                    vel*sp.tan(steer),
                    accel,
                    steer_vel])

    return state_dot


#state and action dimensions
n_x = 10
n_u = 2

#get symbolic variables
state, action = GetSyms(n_x, n_u)

#Construct dynamics
state_dot = sp.Matrix([0.0]*n_x)
# ego vehicle kinematics
state_dot[:5, :] = vehicle_kinematics(state[:5], action)
# other vehicle kinematics (constant velocity and steering)
state_dot[5:, :] = vehicle_kinematics(state[5:], [0, 0])
#construct
dynamics = Dynamics.SymContinuous(state_dot, state, action)


#Construct cost to overtake
px1, py1, heading1, vel1, steer1 = state[:5]
px2, py2, heading2, vel2, steer2 = state[5:]
#cost for reference lane
L = 0.2*(py1 - 1.5)**2
#cost on velocity
L += (vel1*sp.cos(heading1) - 2)**2 + (vel1 - 2)**2
#penality on actions
L += 0.1*action[1]**2 + 0.1*action[0]**2

#collision avoidance (do not cross ellipse around the vehicle)
L += SoftConstrain([((px1 - px2)/4.5)**2 + ((py1 - py2)/2)**2 - 1])
#constrain steering angle and y-position
L += Bounded([py1, steer1], high=[2.5, 0.523], low=[-2.5, -0.523])
#construct
cost = Cost.Symbolic(L, 0, state, action)
#initialise the controller
controller = iLQR(dynamics, cost)
#prediction Horizon
N = 20
#initial state
x0 = np.array([0, 0, 0, 0, 0,
               0, 0, 0, 1, 0])
#initil guess
us_init = np.random.randn(N, n_u)*0.0001
#get optimal states and actions
xs, us, cost_trace = controller.fit(x0, us_init, 100)

#Plot theta and action trajectory
# plt.figure(0)
# theta = np.arctan2(xs[:, 0], xs[:, 1])
# theta = np.where(theta < 0, 2*np.pi+theta, theta)
# plt.plot(theta)
# plt.plot(us)
# plt.plot(cost_trace)

plt.figure(0)
plt.plot(xs[:,0],xs[:,1],label='DDP px-py',linewidth=2.5)
plt.plot(xs[:,5],xs[:,6],label='ref px-py',linewidth=2.5)
# plt.plot(xr[:,0],xr[:,1],label='ref px-py',linewidth=2.5)
plt.legend(fontsize='large', loc='upper right', prop={'family': 'Times New Roman', 'size': 25})
plt.xlabel("px [m]", fontdict={'family': 'Times New Roman', 'size': 20})
plt.ylabel("py [m]", fontdict={'family': 'Times New Roman', 'size': 20})

# plt.show()
plt.figure(1)
plt.plot(xs[:,0],label='LQR px',linewidth=2.5)
plt.plot(xs[:,5],label='ref px',linewidth=2.5)
plt.legend(fontsize='large', loc='upper right', prop={'family': 'Times New Roman', 'size': 25})
plt.xlabel("time [s]", fontdict={'family': 'Times New Roman', 'size': 20})
plt.ylabel("px [m]", fontdict={'family': 'Times New Roman', 'size': 20})

plt.figure(2)
plt.plot(xs[:,1],label='LQR py',linewidth=2.5)
plt.plot(xs[:,6],label='ref py',linewidth=2.5)
plt.legend(fontsize='large', loc='upper right', prop={'family': 'Times New Roman', 'size': 25})
plt.xlabel("time [s]", fontdict={'family': 'Times New Roman', 'size': 20})
plt.ylabel("py [m]", fontdict={'family': 'Times New Roman', 'size': 20})

plt.figure(3)
plt.plot(xs[:,2],label='LQR heading',linewidth=2.5)
plt.plot(xs[:,7],label='ref heading',linewidth=2.5)
plt.legend(fontsize='large', loc='upper right', prop={'family': 'Times New Roman', 'size': 25})
plt.xlabel("time [s]", fontdict={'family': 'Times New Roman', 'size': 20})
plt.ylabel("heading [rad]", fontdict={'family': 'Times New Roman', 'size': 20})

plt.figure(4)
plt.plot(xs[:,3],label='LQR vel',linewidth=2.5)
plt.plot(xs[:,8],label='ref vel',linewidth=2.5)
plt.legend(fontsize='large', loc='upper right', prop={'family': 'Times New Roman', 'size': 25})
plt.xlabel("time [s]", fontdict={'family': 'Times New Roman', 'size': 20})
plt.ylabel("vel [m/s]", fontdict={'family': 'Times New Roman', 'size': 20})

plt.figure(5)
plt.plot(xs[:,4],label='LQR steer',linewidth=2.5)
plt.plot(xs[:,9],label='ref steer',linewidth=2.5)
plt.legend(fontsize='large', loc='upper right', prop={'family': 'Times New Roman', 'size': 25})
plt.xlabel("time [s]", fontdict={'family': 'Times New Roman', 'size': 20})
plt.ylabel("steer [rad]", fontdict={'family': 'Times New Roman', 'size': 20})


plt.figure(6)
plt.plot(us[:,0],label='LQR accel',linewidth=2.5)
plt.legend(fontsize='large', loc='upper right', prop={'family': 'Times New Roman', 'size': 25})
plt.xlabel("time [s]", fontdict={'family': 'Times New Roman', 'size': 20})
plt.ylabel("accel [rad]", fontdict={'family': 'Times New Roman', 'size': 20})

plt.figure(7)
plt.plot(us[:,1],label='LQR steer_vel',linewidth=2.5)
plt.legend(fontsize='large', loc='upper right', prop={'family': 'Times New Roman', 'size': 25})
plt.xlabel("time [s]", fontdict={'family': 'Times New Roman', 'size': 20})
plt.ylabel("steer_vel [rad]", fontdict={'family': 'Times New Roman', 'size': 20})


plt.show()