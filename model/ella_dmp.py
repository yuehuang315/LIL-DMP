import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso,LassoCV,LassoLarsCV   # Lasso回归,LassoCV交叉验证实现alpha的选取，LassoLarsCV基于最小角回归交叉验证实现alpha的选取
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression, Lasso
import ELLAtheta as ll
import readfile as rf
from dmpw import DMPs
import dmpw

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
        return x * (self.goal[dmp_num] - self.y0[dmp_num])

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

# 样本数据集，第一列为x，第二列为y，在x和y之间建立回归模型
files=['0','1','2']
trajectories  = rf.get_sequence(files, writebox=[-1,1,0,1], spaces=False)
trajectories  = trajectories.T
num_DOF = trajectories.shape[0]

breaks = np.array(np.where(trajectories [0] != trajectories [0]))[0]
task_num = len(breaks) - 1
# task_num is the number of characters
task_sets = []
theta = []
psi_sets = []
ella_sets = []

# Generate X, y, and theta
for ii in range(task_num):
    # get the ii'th sequence
    task = trajectories[:, breaks[ii]+1:breaks[ii+1]]
    dmps = DMPs_discrete(n_dmps=num_DOF, n_bfs=100)
#########################################################
################### n_bfs = 1 ###########################
    w, f = dmps.imitate_path(y_des=task)
    # print('w',w)
    # w dimension is n_dmps by n_bfs
    y_track, dy_track, ddy_track, psi, ella = dmps.rollout()
    task_sets.append(task.T)
    # we first caculate x axis!!!!!!!!!!!!!!!!!!!!!!!!
    theta.append(w[0])
    psi_sets.append(psi)
    # Here, the shape of ella is task_num*(x_list_n_bfs,y_list_n_bfs)
    # because ella = psi * w, and w is decided by x and y
    ella_sets.append(ella)
# Here, the shape of theta is task_num*(x_list_n_bfs,y_list_n_bfs)
# print(theta)


# ELLA input X: psi
# ELLA input y: ella
X = []
y = []
ella_y = []
# 生成X和y矩阵
for ii in range(task_num):
    psiMat = np.array(psi_sets[ii])
    ellaMat = np.array(ella_sets[ii])
    X.append(psiMat[:])   #变量x: psi
    # we first caculate x axis!!!!!!!!!!!!!!!!!!!!!!!!
    y.append(ellaMat[:,0])   #变量y: ella_term
    ella_y.append(ellaMat[:,1])
# print('y_x',y)

####????拟合的变量y is f(force) or something about motor control?????
####!!!!拟合的变量y can NOT be trajectories, 
####because trajectory in different tasks is changing
#========Lasso回归========
# model = Lasso(alpha=0.01)  # 调节alpha可以实现对拟合的程度
# model = LassoCV()  # LassoCV自动调节alpha可以实现选择最佳的alpha。
# model = LassoLarsCV()  # LassoLarsCV自动调节alpha可以实现选择最佳的alpha
model = ll.ELLA(d=100, k=100, base_learner=Ridge)
for i in range(task_num):
	#X[i] = X[i].reshape(-1, 1)
	model.fit(X[i], y[i], i, theta[i])	# 线性回归建模
# print('系数矩阵:\n',model.base_learner.coef_)
# print('线性回归模型:\n',model)
# print('最佳的alpha：',model.alpha_)  # 只有在使用LassoCV、LassoLarsCV时才有效
# 使用模型预测 prediction vector is ella_term
predicted = []
for i in range(task_num):
	predicted.append(model.predict(X[i],i))
# print('模型预测:\n',predicted)

# ella_y = np.transpose([ella_y[0]])
# ella_opt = np.hstack((predicted[0],ella_y))
# print(ella_opt)

dmps.reset_state()
imitate_trajectory = []
for ii in range(task_num):
    # get the ii'th sequence
    task_opt = trajectories[:, breaks[ii]+1:breaks[ii+1]]
    dmps = DMPs_discrete(n_dmps=num_DOF, n_bfs=100)
#########################################################
################### n_bfs = 1 ###########################
    w, f = dmps.imitate_path(y_des=task_opt)
    # print('w',w)
    # w dimension is n_dmps by n_bfs
    # ella_y = np.transpose([ella_y[ii]])
    ella_opt = np.hstack((predicted[ii],np.transpose([ella_y[ii]])))
    y_track_opt, dy_track_opt, ddy_track_opt= dmps.rollout_opt(ella_opt)
    imitate_trajectory.append(y_track_opt)
# print(imitate_trajectory)

# 绘制散点图 参数：x横轴 y纵轴
#plt.scatter(X, y, marker='x')
#plt.plot(X, predicted,c='r')
for i in range(task_num):
    a = plt.scatter(task_sets[i][:, 0], task_sets[i][:, 1], c='gray', marker='x')
    plt.legend([a], ['desired path'], loc='upper right')
    plt.plot(imitate_trajectory[i][:,0], imitate_trajectory[i][:,1], c='r')
# for i in range(task_num):
# 	plt.scatter(X[i][:, 0], X[i][:, 1], marker='x')
#     plt.plot(X[i][:, 0],predicted[i],c='r')
#     plt.plot(X[i][:, 0],y[i],c='y')


# 绘制x轴和y轴坐标
plt.xlabel("x")
plt.ylabel("y")


# 显示图形
plt.show()