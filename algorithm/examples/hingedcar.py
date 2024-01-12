'''
 Vehicle Overtaking
 Adjust cost and initial state to get desired behaviors
'''
import sympy as sp
import numpy as np
from ilqr import *
from ilqr.containers import *
import matplotlib.pyplot as plt
import time
lf = 2
lr = 2
# ————————————————获取参考轨迹量——————————————
A=0.06
W=2*np.pi/10
speed=1 #期望速度
dt=0.2
N=50
def vehicle_kinematics(state, action):
    px, py, heading, vel, steer = state
    accel, steer_vel = action
    state_dot = sp.Matrix([
                    px+vel*sp.cos(heading)*dt,
                    py+vel*sp.sin(heading)*dt,
                    heading+(vel * sp.sin(steer) / (lr + lf * sp.cos(steer)) + (steer_vel*lr) / (lr + lf * sp.cos(steer)))*dt,
                    vel+accel*dt,
                    steer+steer_vel*dt])
    return state_dot

def reference_kinematics(state):
    px, py, heading, vel, steer = state
    state_dot = sp.Matrix([
        px+vel*dt,
        py+A * sp.sin( W *px),
        heading+A * W* sp.cos(W *px),
        speed,
        0])
    return state_dot

#state and action dimensions
n_x = 10
n_u = 2
x0 = np.array([0, 0, 0, 0, 0,
               0, 0, 0, 0, 0])
# x0r = np.array([0, 0, 0, 5, 0])
#prediction Horizon
state, action = GetSyms(n_x, n_u)
# state1, action1 = GetSyms(n_x, n_u)
us_init = np.random.randn(N, n_u)*0.0001
state_dot = sp.Matrix([0.0]*n_x)
# state_dot1 = sp.Matrix([0.0]*n_x)
state_dot[:5, :] = vehicle_kinematics(state[:5], action)
# state_dot1[:5, :]= reference_kinematics(x0r)
# state_dot[5:, :] = vehicle_kinematics(state[5:],[0,0])
state_dot[5:, :] = reference_kinematics(state[5:])
dynamics = Dynamics.SymContinuous(state_dot, state, action)

px1, py1, heading1, vel1, steer1 = state[:5]
a1,omiga1=action
px2, py2, heading2, vel2, steer2 = state[5:]
# a2,omiga2=action[2:]
L=50*(px1-px2)**2+50*(py1 - py2)**2+50*(heading1 - heading2)**2+\
  5*(vel1 - vel2)**2+1*(steer1 - steer2)**2+10*(a1)**2+1*(omiga1)**2

# L=(px1-px2)**2+(py1 - py2)**2
# L=0
# L += 0.2*(py1 - 1.5)**2
# # #cost on velocity
# L += (vel1*sp.cos(heading1) - 2)**2 + (vel1 - 2)**2
# # #penality on actions
# L += 0.1*action[1]**2 + 0.1*action[0]**2
# # #collision avoidance (do not cross ellipse around the vehicle)
# L += SoftConstrain([((px1 - px2)/4.5)**2 + ((py1 - py2)/2)**2 - 1])
# # #constrain steering angle and y-position
L += Bounded([a1, omiga1], high=[2, 0.3], low=[-2, -0.3])

#construct
cost = Cost.Symbolic(L, 0, state, action)
#initialise the controller
controller = iLQR(dynamics, cost)
#get optimal states and actions
xs, us, cost_trace = controller.fit(x0, us_init)

# xr=reference_trajectory(0,x0r)

# Plot theta and action trajectory
# plt.figure(0)
# theta = np.arctan2(xs[:, 0], xs[:, 1])
# theta = np.where(theta < 0, 2*np.pi+theta, theta)
# plt.plot(theta)
# plt.plot(us)
# plt.plot(cost_trace)

plt.figure(0)
plt.plot(xs[:,0],xs[:,1],label='iLQR px-py',linewidth=2.5)
plt.plot(xs[:,5],xs[:,6],label='ref px-py',linewidth=2.5)
# plt.plot(np.array(xr)[1:, 0],np.array(xr)[1:, 1],label='LQR px-py',linewidth=2.5)
# plt.plot(xr[:,0],xr[:,1],label='ref px-py',linewidth=2.5)
plt.legend(fontsize='large', loc='upper right', prop={'family': 'Times New Roman', 'size': 25})
plt.xlabel("px [m]", fontdict={'family': 'Times New Roman', 'size': 20})
plt.ylabel("py [m]", fontdict={'family': 'Times New Roman', 'size': 20})

plt.figure(1)
plt.plot(xs[:,0],label='iLQR px',linewidth=2.5)
plt.plot(xs[:,5],label='ref px',linewidth=2.5)
# plt.plot(np.array(xr)[1:, 0],label='ref px',linewidth=2.5)
# plt.plot(xr[:,0],label='ref px',linewidth=2.5)
plt.legend(fontsize='large', loc='upper right', prop={'family': 'Times New Roman', 'size': 25})
plt.xlabel("time [s]", fontdict={'family': 'Times New Roman', 'size': 20})
plt.ylabel("px [m]", fontdict={'family': 'Times New Roman', 'size': 20})

plt.figure(2)
plt.plot(xs[:,1],label='iLQR py',linewidth=2.5)
plt.plot(xs[:,6],label='ref py',linewidth=2.5)
# plt.plot(np.array(xr)[1:, 1],label='ref py',linewidth=2.5)
# plt.plot(xr[:,1],label='ref py',linewidth=2.5)
plt.legend(fontsize='large', loc='upper right', prop={'family': 'Times New Roman', 'size': 25})
plt.xlabel("time [s]", fontdict={'family': 'Times New Roman', 'size': 20})
plt.ylabel("py [m]", fontdict={'family': 'Times New Roman', 'size': 20})

plt.figure(3)
plt.plot(xs[:,2],label='iLQR heading',linewidth=2.5)
plt.plot(xs[:,7],label='ref heading',linewidth=2.5)
# plt.plot(np.array(xr)[1:, 2],label='ref heading',linewidth=2.5)
# plt.plot(xr[:,2],label='ref heading',linewidth=2.5)
plt.legend(fontsize='large', loc='upper right', prop={'family': 'Times New Roman', 'size': 25})
plt.xlabel("time [s]", fontdict={'family': 'Times New Roman', 'size': 20})
plt.ylabel("heading [rad]", fontdict={'family': 'Times New Roman', 'size': 20})

plt.figure(4)
plt.plot(xs[:,3],label='iLQR vel',linewidth=2.5)
plt.plot(xs[:,8],label='ref vel',linewidth=2.5)
# plt.plot(np.array(xr)[1:, 3],label='ref vel',linewidth=2.5)
plt.legend(fontsize='large', loc='upper right', prop={'family': 'Times New Roman', 'size': 25})
plt.xlabel("time [s]", fontdict={'family': 'Times New Roman', 'size': 20})
plt.ylabel("vel [m/s]", fontdict={'family': 'Times New Roman', 'size': 20})

plt.figure(5)
plt.plot(xs[:,4],label='iLQR steer',linewidth=2.5)
plt.plot(xs[:,9],label='ref steer',linewidth=2.5)
# plt.plot(np.array(xr)[1:, 4],label='ref steer',linewidth=2.5)
plt.legend(fontsize='large', loc='upper right', prop={'family': 'Times New Roman', 'size': 25})
plt.xlabel("time [s]", fontdict={'family': 'Times New Roman', 'size': 20})
plt.ylabel("steer [rad]", fontdict={'family': 'Times New Roman', 'size': 20})


plt.figure(6)
plt.plot(us[:,0],label='iLQR accel',linewidth=2.5)
plt.legend(fontsize='large', loc='upper right', prop={'family': 'Times New Roman', 'size': 25})
plt.xlabel("time [s]", fontdict={'family': 'Times New Roman', 'size': 20})
plt.ylabel("accel [rad]", fontdict={'family': 'Times New Roman', 'size': 20})

plt.figure(7)
plt.plot(us[:,1],label='iLQR steer_vel',linewidth=2.5)
plt.legend(fontsize='large', loc='upper right', prop={'family': 'Times New Roman', 'size': 25})
plt.xlabel("time [s]", fontdict={'family': 'Times New Roman', 'size': 20})
plt.ylabel("steer_vel [rad]", fontdict={'family': 'Times New Roman', 'size': 20})


plt.show()
