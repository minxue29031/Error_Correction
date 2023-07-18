import numpy as np

# 加载原始文件
keys = np.load('keys_python_python.npy')
inputs = np.load('inputs_python_python.npy')
values = np.load('values_python_python.npy')

# 计算前三分之一的索引范围
one_third = len(keys) // 3
start_index = 0
end_index = one_third

# 提取前三分之一的数据
keys_subset = keys[start_index:end_index]
inputs_subset = inputs[start_index:end_index]
values_subset = values[start_index:end_index]

# 保存提取的数据到新文件
np.save('keys_python_python2.npy', keys_subset)
np.save('inputs_python_python2.npy', inputs_subset)
np.save('values_python_python2.npy', values_subset)

# 输出新文件的shape
print('keys_python_python2.npy shape:', keys_subset.shape)
print('inputs_python_python2.npy shape:', inputs_subset.shape)
print('values_python_python2.npy shape:', values_subset.shape)