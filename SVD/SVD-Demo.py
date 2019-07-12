'''
# Reconstruct SVD
from numpy import array
from numpy import diag
from numpy import dot
from numpy import zeros
from scipy.linalg import svd
# define a matrix
A = array([[1, 2], [3, 4], [5, 6]])
print(A)
# Singular-value decomposition
U, s, VT = svd(A)
# create m x n Sigma matrix
Sigma = zeros((A.shape[0], A.shape[1]))
# populate Sigma with n x n diagonal matrix
Sigma[:A.shape[1], :A.shape[1]] = diag(s)
print(Sigma)
# reconstruct matrix
B = U.dot(Sigma.dot(VT))
print(B)
exit()
'''

'''
# Singular-value decomposition
from numpy import array
from scipy.linalg import svd
# define a matrix
A = array([[1, 2], [3, 4], [5, 6]])
print(A)
# SVD
U, s, VT = svd(A)
Sigma = zeros((A.shape[0], A.shape[1]))
Sigma[:A.shape[1], :A.shape[1]] = diag(s)
print(U)
print(s)
print(VT)
exit()
'''


import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
from sklearn.preprocessing import normalize

la = np.linalg
words = ["I","like","enjoy","deep","learning","NLP","flying","."]



x = np.array([
	[0,2,1,0,0,0,0,0],
	[2,0,0,1,0,1,0,0],
	[1,0,0,0,0,0,1,0],
	[0,1,0,0,1,0,0,0],
	[0,0,0,1,0,0,0,1],
	[0,1,0,0,0,0,0,1],
	[0,0,1,0,0,0,0,1],
	[0,0,0,0,1,1,1,0]
	])

# x = np.array([
# 	[0,0,1,1,0,0,0,0,0],
# 	[0,0,0,0,0,1,0,0,1],
# 	[0,1,0,0,0,0,0,1,0],
# 	[0,0,0,0,0,0,1,0,1],
# 	[1,0,0,0,0,1,0,0,0],
# 	[1,1,1,1,1,1,1,1,1],
# 	[1,0,1,0,0,0,0,0,0],
# 	[0,0,0,0,0,0,1,0,1],
# 	[0,0,0,0,0,2,0,0,1],
# 	[1,0,1,0,0,0,0,1,0],
# 	[0,0,0,1,1,0,0,0,0],
# 	])

# print(x.ndim)
# exit()


# print(x.ndim - 2)
# exit()
row, column = x.shape # (11,9)
# x_normed = x / x.max(axis=0)
# x = normalize(x, axis=0, norm='max')
# print(x_normed)
# exit()
U, s, Vh = svd(x)

# print(sum(s)*0.9) # 12.65
sig2 = s**2
# print(sum(sig2) * 0.9) # 29.7
print( sum(sig2[:2]) )
exit()
# Sigma = np.zeros((x.shape[0], x.shape[1]))
# Sigma[:x.shape[1], :x.shape[1]] = np.diag(s)
# print(U)
# print('*'*40)
# print(s)
# print('*'*40)
# print(Vh)

U = U*-1
Vh = Vh*-1

# exit()
'''
exit()
with open('matrix.txt', 'w') as f:
	for index, weights in enumerate(U):
		print(index, words[weights.argsort()[-1]], max(weights))
		for weight in weights:
			f.write(str(weight)+"\t")
		f.write("\n")
exit()
'''

for i in range(column):
	plt.text(Vh[3,i], Vh[4,i], 'doc'+str(i))


for i in range(row):
	plt.text(U[i,3], U[i,4], 'word'+str(i))

plt.show()