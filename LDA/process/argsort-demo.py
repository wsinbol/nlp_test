import numpy as np

arr = np.array([8,1,9,2])
print(arr.argsort())
for i in arr.argsort()[:-len(arr)-1:-1]:
	print(i)