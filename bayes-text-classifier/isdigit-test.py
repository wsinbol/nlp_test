
# isdight:检测字符串是否只由数字组成
# isnumeric() 方法检测字符串是否只由数字组成。这种方法是只针对unicode对象。
num = 20 # AttributeError: 'int' object has no attribute 'isdigit'
num = '20' # True
num = '20.1' # 'False'
num = 20.1 # AttributeError: 'float' object has no attribute 'isdigit'
print(num.isdigit())
print()