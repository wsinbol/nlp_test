

num = 20 # AttributeError: 'int' object has no attribute 'isdigit'
num = '20' # True
num = '20.1' # 'False'
# num = 20.1 # AttributeError: 'float' object has no attribute 'isdigit'
print(num.isdigit())