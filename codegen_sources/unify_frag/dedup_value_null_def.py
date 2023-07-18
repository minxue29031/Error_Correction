import numpy as np

# 加载原始文件
values = np.load('values_python_python.npy')
keys = np.load('keys_python_python.npy')
inputs = np.load('inputs_python_python.npy')

# (1) 对values进行去重
values_unique = np.unique(values)
np.save('values_python_python2.npy', values_unique)

# (2) 获取values_unique中每行数据在values中的索引
indices = [np.where(values == value)[0][0] for value in values_unique]

# 根据索引提取对应行的数据
keys_unique = keys[indices]
inputs_unique = inputs[indices]

# 保存结果
np.save('keys_python_python2.npy', keys_unique)
np.save('inputs_python_python2.npy', inputs_unique)

# (3) 返回结果的shape
values_shape = values_unique.shape
keys_shape = keys_unique.shape
inputs_shape = inputs_unique.shape

# 打印结果
print("values_python_python2.npy的shape:", values_shape)
print("keys_python_python2.npy的shape:", keys_shape)
print("inputs_python_python2.npy的shape:", inputs_shape)
