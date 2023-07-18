import numpy as np

# (1) 读取deduplicate_outprefix.tok文件，提取为空或为 "def" 的行的索引
with open("deduplicate_outprefix.tok", "r") as file:
    lines = file.readlines()

empty_def_indices = [index for index, line in enumerate(lines) if line.strip() == "ASTYDJHTGCRTSWRDGFHDXUNGEYCGTEG"]

# 读取keys_python_python.npy、inputs_python_python.npy和values_python_python.npy文件
keys = np.load("keys_python_python.npy")
inputs = np.load("inputs_python_python.npy")
values = np.load("values_python_python.npy")

# 根据空或为 "def" 的行的索引提取对应行生成新的数组，并保存为keys_python_python1.npy、inputs_python_python1.npy和values_python_python1.npy
keys_1 = keys[empty_def_indices]
inputs_1 = inputs[empty_def_indices]
values_1 = values[empty_def_indices]

np.save("keys_python_python1.npy", keys_1)
np.save("inputs_python_python1.npy", inputs_1)
np.save("values_python_python1.npy", values_1)

# (2) 删除deduplicate_outprefix.tok中为空或为 "def" 的行，并删除在其他文件中对应的行
deduplicate_outprefix2 = []
keys_2 = []
inputs_2 = []
values_2 = []

for index, line in enumerate(lines):
    if index not in empty_def_indices:
        deduplicate_outprefix2.append(line.strip())
        keys_2.append(keys[index])
        inputs_2.append(inputs[index])
        values_2.append(values[index])

deduplicate_outprefix2 = "\n".join(deduplicate_outprefix2)

with open("deduplicate_outprefix2.tok", "w") as file:
    file.write(deduplicate_outprefix2)

np.save("keys_python_python2.npy", np.array(keys_2))
np.save("inputs_python_python2.npy", np.array(inputs_2))
np.save("values_python_python2.npy", np.array(values_2))

# (3) 对deduplicate_outprefix2.tok进行去重，保存结果为deduplicate_outprefix3.tok
deduplicate_outprefix2_lines = deduplicate_outprefix2.split("\n")
deduplicate_outprefix3_lines = list(set(deduplicate_outprefix2_lines))
deduplicate_outprefix3 = "\n".join(deduplicate_outprefix3_lines)

with open("deduplicate_outprefix3.tok", "w") as file:
    file.write(deduplicate_outprefix3)

# 返回deduplicate_outprefix3.tok中每行数据在deduplicate_outprefix2.tok中的索引
indices_3 = [deduplicate_outprefix2_lines.index(line) for line in deduplicate_outprefix3_lines]

# 从keys_python_python2.npy、inputs_python_python2.npy和values_python_python2.npy中提取对应行的数据
keys_3 = np.array(keys_2)[indices_3]
inputs_3 = np.array(inputs_2)[indices_3]
values_3 = np.array(values_2)[indices_3]

np.save("keys_python_python3.npy", keys_3)
np.save("inputs_python_python3.npy", inputs_3)
np.save("values_python_python3.npy", values_3)


# (4) 获取所有123.npy文件的形状
shape_keys_1 = np.load("keys_python_python1.npy").shape
shape_inputs_1 = np.load("inputs_python_python1.npy").shape
shape_values_1 = np.load("values_python_python1.npy").shape

shape_keys_2 = np.load("keys_python_python2.npy").shape
shape_inputs_2 = np.load("inputs_python_python2.npy").shape
shape_values_2 = np.load("values_python_python2.npy").shape

shape_keys_3 = np.load("keys_python_python3.npy").shape
shape_inputs_3 = np.load("inputs_python_python3.npy").shape
shape_values_3 = np.load("values_python_python3.npy").shape

print("Shape of keys_python_python1.npy:", shape_keys_1)
print("Shape of inputs_python_python1.npy:", shape_inputs_1)
print("Shape of values_python_python1.npy:", shape_values_1)

print("Shape of keys_python_python2.npy:", shape_keys_2)
print("Shape of inputs_python_python2.npy:", shape_inputs_2)
print("Shape of values_python_python2.npy:", shape_values_2)

print("Shape of keys_python_python3.npy:", shape_keys_3)
print("Shape of inputs_python_python3.npy:", shape_inputs_3)
print("Shape of values_python_python3.npy:", shape_values_3)
