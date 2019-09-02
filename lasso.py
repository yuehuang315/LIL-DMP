import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso,LassoCV,LassoLarsCV   # Lasso回归,LassoCV交叉验证实现alpha的选取，LassoLarsCV基于最小角回归交叉验证实现alpha的选取
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression, Lasso
import ELLA as ll
import readfile as rf

# 样本数据集，第一列为x，第二列为y，在x和y之间建立回归模型
# files=['0','1','2']
files=['0']
nums = rf.get_sequence(files, writebox=[-1,1,0,1], spaces=False)
nums = nums[1:-1,:]

#生成X和y矩阵
dataMat = np.array(nums)
# X = dataMat[:, 0]   #变量x
# X = X.reshape(-1, 1)
X = dataMat[:]   #变量x
y = dataMat[:, 1]   #变量y
# print('模型y:\n',y)
print('模型X:\n',X.shape)


#####????拟合的变量y is f(force) or something about motor control?????
#####because trajectory in different tasks is changing
# ========Lasso回归========
# model = Lasso(alpha=0.01)  # 调节alpha可以实现对拟合的程度
# model = LassoCV()  # LassoCV自动调节alpha可以实现选择最佳的alpha。
# model = LassoLarsCV()  # LassoLarsCV自动调节alpha可以实现选择最佳的alpha
model = ll.ELLA(d=2, k=1, base_learner=Ridge)
#for i in range task_id:
#	model.fit(X[i], y[i], i)	# 线性回归建模  
model.fit(X, y, 0)	# 线性回归建模 
#print('系数矩阵:\n',model.base_learner.coef_)
print('线性回归模型:\n',model)
# print('最佳的alpha：',model.alpha_)  # 只有在使用LassoCV、LassoLarsCV时才有效
# 使用模型预测
predicted = model.predict(X,0)
print('模型预测:\n',predicted)

# 绘制散点图 参数：x横轴 y纵轴
#plt.scatter(X, y, marker='x')
#plt.plot(X, predicted,c='r')
plt.scatter(X[:,0], y, marker='x')
plt.plot(X[:,0],predicted*2,c='r')


# 绘制x轴和y轴坐标
plt.xlabel("x")
plt.ylabel("y")

# 显示图形
plt.show()