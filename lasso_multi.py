import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso,LassoCV,LassoLarsCV   # Lasso回归,LassoCV交叉验证实现alpha的选取，LassoLarsCV基于最小角回归交叉验证实现alpha的选取
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression, Lasso
import ELLA as ll
import readfile as rf

# 样本数据集，第一列为x，第二列为y，在x和y之间建立回归模型
files=['0','1','2','3','4','5','6','7','8','9']
nums = rf.get_sequence(files, writebox=[-1,1,0,1], spaces=False)
nums = nums.T

breaks = np.array(np.where(nums[0] != nums[0]))[0]
task_num = len(breaks) - 1
num_sets = []
# task_num is the number of characters
for ii in range(task_num):
    # get the ii'th sequence
    num = nums[:, breaks[ii]+1:breaks[ii+1]]
    num_sets.append(num)

X = []
y = []
#生成X和y矩阵
for ii in range(task_num):
	dataMat = np.array(num_sets[ii])
	dataMat = dataMat.T
	X.append(dataMat[:])   #变量x
	y.append(dataMat[:, 1])   #变量y


####????拟合的变量y is f(force) or something about motor control?????
####because trajectory in different tasks is changing
#========Lasso回归========
# model = Lasso(alpha=0.01)  # 调节alpha可以实现对拟合的程度
# model = LassoCV()  # LassoCV自动调节alpha可以实现选择最佳的alpha。
# model = LassoLarsCV()  # LassoLarsCV自动调节alpha可以实现选择最佳的alpha
model = ll.ELLA(d=2, k=1, base_learner=Ridge)
for i in range(task_num):
	#X[i] = X[i].reshape(-1, 1)
	model.fit(X[i], y[i], i)	# 线性回归建模  
# print('系数矩阵:\n',model.base_learner.coef_)
# print('线性回归模型:\n',model)
# print('最佳的alpha：',model.alpha_)  # 只有在使用LassoCV、LassoLarsCV时才有效
# 使用模型预测
predicted = []
for i in range(task_num):
	predicted.append(model.predict(X[i],i))

# print('模型预测:\n',predicted)

# 绘制散点图 参数：x横轴 y纵轴
# plt.scatter(X, y, marker='x')
# plt.plot(X, predicted,c='r')
for i in range(task_num):
	a = plt.scatter(X[i][:, 0], y[i], c='gray', marker='x')
	plt.legend([a], ['desired path'], loc='upper right')
	plt.plot(X[i][:, 0],predicted[i],c='r')


# 绘制x轴和y轴坐标
plt.xlabel("x")
plt.ylabel("y")


# 显示图形
plt.show()