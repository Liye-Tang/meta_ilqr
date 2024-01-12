import casadi as ca
import numpy as np
import math
import time
import matplotlib.pyplot as plt
import pandas as pd

# the vehicle params
lf = 2  # m
lr = 2 # m
T = 0.1  # time step
N = 20  # predict hoziron length
# weight matrix
Q = np.diag([5000, 50000, 50000, 0, 10]) #state
R = np.diag([1000, 10])
init_state = np.array([0.0, 0.0, 0.0, 0, 0.0])# init_state   x, y, phi, v, theta
#max simulation step
step=200

A=1.5
W=2*np.pi/10
R=0

speed=5 #期望速度

# ————————————状态更新————————————
# def predict_state(x0, u):
#     states = np.zeros((N + 1, 5)) #
#     states[0, :] = x0
#     # euler method
#     for i in range(N):
#         states[i + 1, 0] = states[i, 0] + (T * states[i, 3] * np.cos(states[i, 2]))  # "x"
#         states[i + 1, 1] = states[i, 1] + (T * states[i, 3] * np.sin(states[i, 2]))  # "u"
#         states[i + 1, 2] = states[i, 1] + (T * states[i, 3] + np.sin(states[i, 4])) / (lr + lf * np.cos(states[i, 4])) + (
#                 T * u[i, 1] * lr) / (lr + lf * np.cos(states[i, 4]))  # "phi"
#         states[i + 1, 3] = states[i, 3] + T * (u[i, 0])  # v
#         states[i + 1, 4] = states[i, 4] + T * (u[i, 1])  # theta
#     return states

# ————————————————获取参考轨迹量——————————————
def reference_trajectory(t, x0):
    # initial state / last state
    x_ = np.zeros((N + 1, 5))
    x_[0] = x0
    # states for the next N trajectories
    for i in range(N):
        t_predict = t + T * i  #
        #x_ref=-1 / W*np.cos(W * t_predict + R)+5*t_predict+ 1 / W*np.cos(R)
        x_ref = t_predict * speed
        y_ref = A * np.sin(W * t_predict + R)
        phi_ref = A * W * np.cos(W * t_predict+ R)
        v_ref = speed
        theta_ref = 0
        x_[i + 1] = np.array([x_ref, y_ref, phi_ref, v_ref,theta_ref])

    # for i in range(N):
    #         t_predict = t + T * i  #
    #         x_ref = t_predict*1
    #
    #         if  x_ref<10:
    #             y_ref = 0
    #         elif  10<x_ref<30:
    #             y_ref = 2
    #         elif  30<x_ref<50:
    #             y_ref = 6
    #         else:
    #             y_ref = 2
    #
    #         phi_ref = 0
    #         v_ref = 0
    #         theta_ref = 0
    #         x_[i + 1] = np.array([x_ref, y_ref, phi_ref, v_ref, theta_ref])
    #

    return x_#,u_

# ——————————大循环——————————————
def shift(t0, x0, u, x_n, f):
    f = f(x0, u[0])
    state_curr = x0 + T * f #
    t = t0 + T
    u0 = np.concatenate((u[1:], u[-1:]))
    state_next = np.concatenate((x_n[1:], x_n[-1:]))
    return t, state_curr, u0, state_next  #  t0, current_state, u0, next_states

if __name__ == "__main__":
    opti = ca.Opti()
    opt_x_ref = opti.parameter(N + 1, 5)
    x_ref = opt_x_ref[:, 0]
    y_ref = opt_x_ref[:, 1]
    phi_ref = opt_x_ref[:, 2]
    v_ref = opt_x_ref[:, 3]
    theta_ref = opt_x_ref[:, 4]
    opt_controls = opti.variable(N, 2)
    a = opt_controls[:, 0]  # phi
    omega = opt_controls[:, 1]  # theta

    # state variable: configuration
    opt_states = opti.variable(N + 1, 5)
    x = opt_states[:, 0]
    y = opt_states[:, 1]
    phi = opt_states[:, 2]
    v = opt_states[:, 3]
    theta = opt_states[:, 4]

    # create model
    f = lambda x_, u_: ca.vertcat(      #列向拼接
        x_[3] * ca.cos(x_[2]),
        x_[3] * ca.sin(x_[2]),
        (x_[3] * ca.sin(x_[4])) / (lr + lf * ca.cos(x_[4])) + (u_[1] * lr) / (lr + lf * ca.cos(x_[4])),
        u_[0],
        u_[1],
    )
    f_np = lambda x_, u_:np.array([   #vertcat` 函数返回的是一个 `casadi` 类型的向量，因此需要将其转换为 NumPy 数组才能进行后续计算。
        x_[3] * ca.cos(x_[2]),
        x_[3] * ca.sin(x_[2]),
        (x_[3] *ca.sin(x_[4])) / (lr + lf * ca.cos(x_[4])) + (u_[1] * lr) / (lr + lf * ca.cos(x_[4])),
        u_[0],
        u_[1],
    ])
    # parameters, these parameters are the reference trajectories of the pose and inputs
    #opt_u_ref = opti.parameter(N, 2)


    ##模型约束+离散
    opti.subject_to(opt_states[0, :] == opt_x_ref[0, :])
    for i in range(N):
        x_next = opt_states[i, :] + f(opt_states[i, :], opt_controls[i, :]).T * T  #状态转移，对连续方程进行了离散
        opti.subject_to(opt_states[i + 1, :] == x_next)#将动态约束条件加入到优化问题中

    # cost function
    obj = 0
    for i in range(N):
        state_error_ = opt_states[i, :] - opt_x_ref[i +1, :]
        control_error_ = opt_controls[i, :]
        obj = obj + ca.mtimes([state_error_, Q, state_error_.T]) + ca.mtimes([control_error_, R, control_error_.T])#.T转置矩阵
    #     #print(opt_states[i, 1:])
    # Q = [1, 50, 20, 1, 1]
    # R = [1, 10]
    # for i in range(N):
    #     j = Q[0] * (opt_states[i, 0] - opt_x_ref[i+1, 0]) ** 2 \
    #        + Q[1] * (opt_states[i, 1] - opt_x_ref[i+1, 1]) ** 2 \
    #        + Q[2] * (opt_states[i, 2] - opt_x_ref[i+1, 2]) ** 2 \
    #        + Q[3] * (opt_states[i, 3] - opt_x_ref[i+1, 3]) ** 2 \
    #        + Q[4] * (opt_states[i, 4]) ** 2 \
    #        + R[0] * (opt_controls[i, 0] ** 2) \
    #        + R[1] * (opt_controls[i, 1]) ** 2
    #     obj = obj +j
             # State constraint
    opti.subject_to(opti.bounded(0, x, np.inf))
    opti.subject_to(opti.bounded(-10, y, 10))
    opti.subject_to(opti.bounded(-np.inf, phi, np.inf))
    opti.subject_to(opti.bounded(0, v, 10))
    opti.subject_to(opti.bounded(-np.inf, theta, np.inf))

    # control constraint
    opti.subject_to(opti.bounded(-5, a,5))
    opti.subject_to(opti.bounded(-10, omega, 10))

    # ipopt parameters
    opts_setting = {'ipopt.max_iter': 1e8,
                    'ipopt.print_level': 0,
                    'print_time': 0,
                    'ipopt.acceptable_tol': 1e-4,
                    'ipopt.acceptable_obj_change_tol': 1e-4}
    opti.solver('ipopt', opts_setting)

    t0 = 0
    current_state = init_state
    u0 = np.zeros((N, 2))
    next_trajectories = np.tile(init_state, N + 1).reshape(N + 1,-1)  # 将init_state复制N+1次，N + 1*5列的数组
    next_controls = np.zeros((N, 2))
    next_states = np.zeros((N + 1, 5))
    x_c = []  # contains for the history of the state
    u_c = []
    # print("kongzhi:",u0)
    t_c = [t0]  # for the time
    xx = [current_state]
    #xr = [next_trajectories[0]]
    xr = [init_state]

    ## start MPC
    mpciter = 0
    index_t = []
    opti.minimize(obj)

    for j in range(step):
        #将一个预测时域内的参考量，控制量，状态量的未知量矩阵传入casadi
        opti.set_value(opt_x_ref, next_trajectories)# 将变量opt_x_ref的值设置为next_trajectories，目的是将next_trajectories的值作为数值优化问题的参考轨迹
        opti.set_initial(opt_controls, u0)  # (N, 3)#将变量opt_controls的初始值设置为u0.reshape(N, 2)。这个操作的目的是将控制变量u0的值作为数值优化问题的初始值
        opti.set_initial(opt_states, next_states)  # (N+1, 6)
        dt = time.time()
        sol = opti.solve()
        tt=(time.time() - dt)*1000
        index_t.append(tt) #Single step calculation time
        u_res = sol.value(opt_controls)#使用数值优化器sol计算优化变量opt_controls的最优解，并将最优解存储在变量u_res中
        x_m = sol.value(opt_states)#使用数值优化器sol计算状态变量opt_states的最优解，并将最优解存储在变量x_m中
        u_c.append(u_res[0, :])#取出第一组控制序列作为执行量
        t_c.append(t0)
        t0, current_state, u0, next_states = shift(t0, current_state, u_res, x_m, f_np)
        xx.append(current_state) #
        next_trajectories = reference_trajectory(t0, current_state)
        xr.append(next_trajectories[1])
        print("iteration step:",mpciter+1)
        print("time of every step:", tt, "ms")  #########
        mpciter = mpciter + 1


    # #t_v = np.array(index_t)
    # print(len(index_t),index_t)
    #
    # # 读取CSV文件中原有数据
    # df = pd.read_csv('MPC time.csv')
    #
    # # 添加新行
    # df.loc[len(df)] =index_t
    #
    # # 将更新后的内容写回CSV文件
    # df.to_csv('MPC time.csv', index=False)

    # with open('MPC time.csv', 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(index_t)

    #my_list = xx.flatten().tolist()
    plt.figure(1)
    plt.plot(t_c[1:], np.array(xr)[1:, 0], label='ref x',linewidth=2.5)
    plt.plot(t_c, np.array(xx)[:, 0], label='MPC x',linewidth=2.5)
    plt.legend(fontsize='large', loc='upper right',prop={'family' : 'Times New Roman', 'size' : 25})
    plt.xlabel("time [s]",fontdict={'family' : 'Times New Roman', 'size' : 20})
    plt.ylabel("x [m]",fontdict={'family' : 'Times New Roman', 'size' : 20})
    #plt.grid(True)
    #plt.show()

    plt.figure(2)
    plt.plot(t_c[1:], np.array(xr)[1:, 1], label='ref y',linewidth=2.5)
    plt.plot(t_c, np.array(xx)[:, 1], label='MPC y',linewidth=2.5)
    plt.legend(fontsize='large', loc='upper right', prop={'family': 'Times New Roman', 'size': 25})
    plt.xlabel("time [s]",fontdict={'family' : 'Times New Roman', 'size' : 20})
    plt.ylabel("y [m]",fontdict={'family' : 'Times New Roman', 'size' : 20})
   # plt.grid(True)
    plt.ylim(-2,10)
    #plt.show()

    plt.figure(3)
    plt.plot(t_c[1:], np.array(xr)[1:, 2], label='ref phi',linewidth=2.5)
    plt.plot(t_c, np.array(xx)[:, 2], label='MPC phi',linewidth=2.5)
    plt.legend(fontsize='large', loc='upper right', prop={'family': 'Times New Roman', 'size': 25})
    plt.xlabel("time [s]",fontdict={'family' : 'Times New Roman', 'size' : 20})
    plt.ylabel("phi [rad]",fontdict={'family' : 'Times New Roman', 'size' : 20})
    #plt.grid(True)
    #plt.show()

    plt.figure(4)
    plt.plot(t_c[1:], np.array(xr)[1:, 3], label='ref V',linewidth=2.5)
    plt.plot(t_c, np.array(xx)[:, 3], label='MPC V',linewidth=2.5)
    plt.legend(fontsize='large', loc='upper right', prop={'family': 'Times New Roman', 'size': 25})
    plt.xlabel("time [s]",fontdict={'family' : 'Times New Roman', 'size' : 20})
    plt.ylabel("speed [m/s]",fontdict={'family' : 'Times New Roman', 'size' : 20})
    #plt.grid(True)
    #plt.show()

    plt.figure(5)
   # plt.plot(t_c, np.array(xr)[:, 4], label='ref V')
    plt.plot(t_c, np.array(xx)[:, 4], label='MPC theta',linewidth=2.5)
    plt.legend(fontsize='large', loc='upper right', prop={'family': 'Times New Roman', 'size': 25})
    plt.xlabel("time [s]", fontdict={'family': 'Times New Roman', 'size': 20})
    plt.ylabel("theta [rad]", fontdict={'family': 'Times New Roman', 'size': 20})
    #plt.grid(True)
    plt.show()
# 绘制曲线图
# plt.plot(x, y1, 'dodgerblue', label='actual',linewidth=2.5)
# plt.plot(x, y2, 'darkorange', label='ref',linewidth=2.5)
#
# plt.legend(prop={'family' : 'Times New Roman', 'size' : 25})#fontsize=15      # 设置图例位置  ,prop ='Times New Roman
# #plt.tight_layout()
# plt.xlabel("Time [s]",fontdict={'family' : 'Times New Roman', 'size' : 20}) # x轴名称,fontsize=15
# plt.ylabel("x [m]",fontdict={'family' : 'Times New Roman', 'size' : 20}) #y轴名称,fontsize=15
# plt.tick_params(labelsize=20)



