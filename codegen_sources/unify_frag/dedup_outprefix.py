import numpy as np


with open("deduplicate_outprefix.tok", "r") as file:
    lines = file.readlines()

empty_def_indices = [index for index, line in enumerate(lines) if line.strip() == "" or line.strip() == "def"]

# 读取keys_python_python.npy、inputs_python_python.npy和values_python_python.npy文件
keys = np.load("keys_python_python.npy")
inputs = np.load("inputs_python_python.npy")
values = np.load("values_python_python.npy")

keys_1 = keys[empty_def_indices]
inputs_1 = inputs[empty_def_indices]
values_1 = values[empty_def_indices]

np.save("keys_python_python1.npy", keys_1)
np.save("inputs_python_python1.npy", inputs_1)
np.save("values_python_python1.npy", values_1)

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

deduplicate_outprefix2_lines = deduplicate_outprefix2.split("\n")
deduplicate_outprefix3_lines = list(set(deduplicate_outprefix2_lines))
deduplicate_outprefix3 = "\n".join(deduplicate_outprefix3_lines)

with open("deduplicate_outprefix3.tok", "w") as file:
    file.write(deduplicate_outprefix3)

indices_3 = [deduplicate_outprefix2_lines.index(line) for line in deduplicate_outprefix3_lines]

keys_3 = np.array(keys_2)[indices_3]
inputs_3 = np.array(inputs_2)[indices_3]
values_3 = np.array(values_2)[indices_3]

np.save("keys_python_python3.npy", keys_3)
np.save("inputs_python_python3.npy", inputs_3)
np.save("values_python_python3.npy", values_3)


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

