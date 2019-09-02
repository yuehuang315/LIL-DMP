import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import Lasso,LassoCV,LassoLarsCV   # Lasso回归,LassoCV交叉验证实现alpha的选取，LassoLarsCV基于最小角回归交叉验证实现alpha的选取
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression, Lasso
import ELLA as ll
import readfile3d as rf
from dmpw import DMPs
import dmpw
import time

############# DMP Model ###############
class DMPs_discrete(DMPs):
    '''
    An implementation of discrete DMPs
    '''

    def __init__(self, **kwargs):
        # call super class constructor
        super(DMPs_discrete, self).__init__(pattern='discrete', **kwargs)

        self.gen_centers()

        # set variance (h) of Gaussian basis functions
        # trial and error to find this spacing???
        self.h = np.ones(self.n_bfs) * self.n_bfs**1.5 / self.c / self.cs.ax

        self.check_offset()

    def gen_centers(self):
        '''
        Calculate ci
        Set the center of the Gaussian basis
        function be spaced evenly throughout run time
        :return:
        '''

        # desired activations throughout time
        des_c = np.linspace(0, self.cs.run_time, self.n_bfs)
        self.c = np.ones(len(des_c))

        for n in range(len(des_c)):
            self.c[n] = np.exp(-self.cs.ax * des_c[n])

    def gen_front_term(self, x, dmp_num):
        '''
        Generates the diminishing front term on the force term.
        :param x: float, the current value of the canonical system
        :param dmp_num: int, the index of the current dmp
        :return: x(g-y0)
        '''
        return (self.goal[dmp_num] - self.y0[dmp_num])

    def gen_goal(self, y_des):
        '''
        Generate the goal for path imitation
        :param y_des: np.array, the desired trajectory to follow
        :return:
        '''
        return np.copy(self.y_des[:, -1])

    def gen_psi(self, x):
        '''?????????????????????????? x = x[:, None] what????????
        Generate the basis functions for a given canonical system rollout x
        :param x:
        :return: phy function
        '''
        if isinstance(x, np.ndarray):
            x = x[:, None]
        return np.exp(-self.h * (x - self.c)**2)

    def gen_weights(self, f_target):
        '''
        Generate a set of weight over the basis functions
        such that the target forcing term trajectory is matched
        :param f_target:
        '''

        # calculate x and psi
        x_track = self.cs.rollout()
        psi_track = self.gen_psi(x_track)

        # calculate BF weights using weighted linear regression
        self.w = np.zeros((self.n_dmps, self.n_bfs))
        for d in range(self.n_dmps):
            # spatial scaling term
            k = (self.goal[d] - self.y0[d])
            for b in range(self.n_bfs):
                numer = np.sum(x_track * psi_track[:, b] * f_target[:, d] )
                denom = np.sum(x_track**2 * psi_track[:, b])
                self.w[d, b] = numer / (k * denom)
        self.w = np.nan_to_num(self.w)
        return self.w

t_start = time.time()
# 样本数据集
files = ['ur_long_demo1','ur_long_demo2','3d','ur_long_demo1','ur_long_demo2',
'ur_short_demo1','ur_short_demo2','ur_short_demo3','ur_short_demo4','ur_short_demo5',
'ur_short_demo6','ur_short_demo7','ur_short_demo8','ur_short_demo9','ur_short_demo10',
'ur_short_demo11','ur_short_demo12','ur_short_demo13','ur_short_demo14','ur_short_demo15',
'ur_short_demo16','ur_short_demo17','ur_short_demo18','3d','ur_long_demo1',
'B0','B1','B2','B3','B4','B5','B6','B7','B8','B9',
'B10','B11','B12','B13','B14','B15','B16','B17','B18','B19',
's5','s6','s7','s8','ur_long_demo1']
# files = ['s0','s1','s2','s3','s4','s5','s6','s7','s8','s9']

trajectories  = rf.get_sequence(files, writebox=[-1,1,0,1], spaces=False)
# trajectories  = rf.get_sequence(files, writebox = [-.1, .1, .2, .25], spaces=False)
trajectories  = trajectories.T
num_DOF = trajectories.shape[0]
# print("num_DOF",num_DOF)

breaks = np.array(np.where(trajectories [0] != trajectories [0]))[0]
task_num = len(breaks) - 1
print("task_num",task_num)
# task_num is the number of characters

task_sets = []
theta_x = []
theta_y = []
theta_z = []
psi_sets = []
ella_sets = []

# Generate X, y, and theta
for ii in range(task_num):
    # get the ii'th sequence
    task = trajectories[:, breaks[ii]+1:breaks[ii+1]]
    dmps = DMPs_discrete(n_dmps=num_DOF, n_bfs=100)
#########################################################
    w, f = dmps.imitate_path(y_des=task, goal_bias=0)
    # print('w',w)
    # w dimension is n_dmps by n_bfs
    y_track, dy_track, ddy_track, psi, ella = dmps.rollout()
    task_sets.append(task.T)
    # we divide caculate x and y axis!!!!!!!!!!!!!!!!!!!!!!!!
    theta_x.append(w[0])
    theta_y.append(w[1])
    theta_z.append(w[2])
    psi_sets.append(psi)
    # print("psi",len(psi))
    # Here, the shape of ella is task_num*(x_list_n_bfs,y_list_n_bfs)
    # because ella = psi * w, and w is decided by x and y
    ella_sets.append(ella)
# Here, the shape of theta is task_num*(x_list_n_bfs,y_list_n_bfs)

# # ELLA input X: psi, the same between x and y
# # ELLA input y: ella
# X = []
# y_x = []
# y_y = []
# y_z = []
# # 生成X和y矩阵
# for ii in range(task_num):
#     psiMat = np.array(psi_sets[ii])
#     ellaMat = np.array(ella_sets[ii])
#     X.append(psiMat[:])   #变量x: psi
#     # Now we caculate x and y axis!!!!!!!!!!!!!!!!!!!!!!!!
#     y_x.append(ellaMat[:,0])   #变量y: ella_term
#     y_y.append(ellaMat[:,1])   #变量y: ella_term
#     y_z.append(ellaMat[:,2])   #变量y: ella_term
# # print('y_z',y_z)
# print('模型X数目:\n',len(X))

# #========Lasso回归========
# # model = Lasso(alpha=0.01)  # 调节alpha可以实现对拟合的程度
# # model = LassoCV()  # LassoCV自动调节alpha可以实现选择最佳的alpha。
# # model = LassoLarsCV()  # LassoLarsCV自动调节alpha可以实现选择最佳的alpha

# model_x = ll.ELLA(d=100, k=10, base_learner=LinearRegression)
# model_y = ll.ELLA(d=100, k=10, base_learner=Ridge)
# model_z = ll.ELLA(d=100, k=10, base_learner=Ridge)
# for i in range(task_num):
#     # model_x.fit(X[i], y_x[i], i, theta_x[i]) # 线性回归建模
#     model_x.fit(X[i], y_x[i], i) # 线性回归建模
#     model_y.fit(X[i], y_y[i], i) # 线性回归建模
#     model_z.fit(X[i], y_z[i], i) # 线性回归建模
# # # print('系数矩阵:\n',s)
# # # print('线性回归模型:\n',model)
# # # print('最佳的alpha：',model.alpha_)  # 只有在使用LassoCV、LassoLarsCV时才有效

# # 使用模型预测 prediction vector is ella_term
# predicted_x = []
# predicted_y = []
# predicted_z = []
# for i in range(task_num):
#     predicted_x.append(model_x.predict(X[i], i))
#     predicted_y.append(model_y.predict(X[i], i))
#     predicted_z.append(model_z.predict(X[i], i))
# # print('模型预测:\n',predicted)

# ella_opt = []
# for ii in range(task_num):
#     ella_opt.append(np.hstack((predicted_x[ii],predicted_y[ii],predicted_z[ii])))

dmps.reset_state()
imitate_task = []
for ii in range(task_num):
    # get the ii'th sequence
    task_opt = trajectories[:, breaks[ii]+1:breaks[ii+1]]
    dmps = DMPs_discrete(n_dmps=num_DOF, n_bfs=100)
    imitate_trajectory = []
#########################################################
################### n_bfs = 1 ###########################
    for iii in range(3):
        w, f = dmps.imitate_path(y_des=task_opt, goal_bias=(iii-1))
        # print('w',w)
        # w dimension is n_dmps by n_bfs
        # ella_y = np.transpose([ella_y[ii]])
        # print('ella_opt',ella_opt)
        y_track_opt, dy_track_opt, ddy_track_opt= dmps.rollout_opt(ella_sets[ii])
        imitate_trajectory.append(y_track_opt)
    imitate_task.append(imitate_trajectory)
###########################################
#######distance############################
###########################################
for ii in range(10):
    print("imitate_task%d:" %ii, imitate_task[ii+30][1])


t_end = time.time()
print("run_time",t_end-t_start)

# ##################################################
# ########one plot##################################
# ##################################################
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# for i in range(task_num - 1):
#     ax.scatter(task_sets[i][:, 0], task_sets[i][:, 1], task_sets[i][:, 2], c='gray', marker='x')
#     ax.plot(imitate_task[i][1][:, 0], imitate_task[i][1][:, 1], imitate_task[i][1][:,2], c='r')
#     # ax.plot(imitate_task[i][0][:, 0], imitate_task[i][0][:, 1], imitate_task[i][0][:,2], c='b')
#     # ax.plot(imitate_task[i][2][:, 0], imitate_task[i][2][:, 1], imitate_task[i][2][:,2], c='b')
    
# ax.scatter(task_sets[task_num-1][:, 0], task_sets[task_num-1][:, 1], task_sets[task_num-1][:, 2], c='gray', marker='x', label='Demo_Path')
# ax.plot(imitate_task[task_num-1][1][:, 0], imitate_task[task_num-1][1][:, 1], imitate_task[task_num-1][1][:,2], c='r', label='Imitate_Path')
# # ax.plot(imitate_task[task_num-1][0][:, 0], imitate_task[task_num-1][0][:, 1], imitate_task[task_num-1][0][:,2], c='b', label='Generalize_Path1')
# # ax.plot(imitate_task[task_num-1][2][:, 0], imitate_task[task_num-1][2][:, 1], imitate_task[task_num-1][2][:,2], c='b', label='Generalize_Path2')
# ax.legend()
# plt.show()

# # 绘制x轴和y轴坐标
# plt.xlabel("x")
# plt.ylabel("y")

# for i in range(task_num):
#     fig = plt.figure()
#     ax = fig.gca(projection='3d')
#     ax.scatter(task_sets[i][:, 0], task_sets[i][:, 1], task_sets[i][:, 2], c='gray', marker='x', label='Demo_Path')
#     ax.plot(imitate_task[i][1][:, 0], imitate_task[i][1][:, 1], imitate_task[i][1][:,2], c='r', label='Imitate_Path')
#     # ax.plot(imitate_task[i][0][:, 0], imitate_task[i][0][:, 1], imitate_task[i][0][:,2], c='b', label='Generalize_Path1')
#     # ax.plot(imitate_task[i][2][:, 0], imitate_task[i][2][:, 1], imitate_task[i][2][:,2], c='b', label='Generalize_Path2')
#     ax.legend()

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.scatter(task_sets[0][:, 0], task_sets[0][:, 1], task_sets[0][:, 2], c='gray', marker='x', label='Demo_Path')
# ax.plot(imitate_task[0][1][:, 0], imitate_task[0][1][:, 1], imitate_task[0][1][:,2], c='r', label='Imitate_Path')
# # ax.plot(imitate_task[i][0][:, 0], imitate_task[i][0][:, 1], imitate_task[i][0][:,2], c='b', label='Generalize_Path1')
# # ax.plot(imitate_task[i][2][:, 0], imitate_task[i][2][:, 1], imitate_task[i][2][:,2], c='b', label='Generalize_Path2')
# ax.legend()

plt.show()
# 绘制x轴和y轴坐标
plt.xlabel("x")
plt.ylabel("y")