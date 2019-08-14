import numpy as np

arr = np.array([8,1,9,2])
# print(arr.argsort())
for i in arr.argsort()[:-len(arr)-1:-1]:
	pass
	# print(i)

a = [i for i in range(10)]
print(a)
for i in a[1:-6:-1]:
	print(i)

for i in a[0:-6:1]:
	print(i)

# for i in a[-10:9]:
# 	print(i)