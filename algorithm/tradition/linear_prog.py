#导入包
from scipy import optimize
import numpy as np

#确定c,A,b,Aeq,beq
c = np.array([2,3,-5])
A = np.array([[-2,5,-1],[1,3,1]])
b = np.array([-10,12])
Aeq = np.array([[1,1,1]])
beq = np.array([7])

#求解
res = optimize.linprog(-c,A,b,Aeq,beq)
print(res)


'''
要解决的问题

MAX z = 2 x1 + 3 x2 - 5x3

st条件:
x1 + x2 + x3 = 7
2 x1 - 5 x2 + x3 >= 10
x1 + 3 x2 + x3 < 12 

解决思路：
调用sci包需要用向量的方式解决表达规划矩阵，需要表达成的matlab标准形式：
1） 目标函数必须是MIN
2） 条件函数必须是 <=
否则需要做代数转化 

'''


