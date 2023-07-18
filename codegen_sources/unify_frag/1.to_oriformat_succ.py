from codegen_sources.preprocessing.lang_processors.python_processor import PythonProcessor
import os

# 创建PythonProcessor实例
processor = PythonProcessor()

# 读取input.tok文件内容
with open('data/ec_result_tcst/succori.tok', 'r') as file:
    lines = file.readlines()

# 创建目录unify（如果不存在）
output_dir = 'data/ec_result_tcst/unify/detokenized'
os.makedirs(output_dir, exist_ok=True)

# 逐行处理和写入输出文件
for i, line in enumerate(lines):
    # 还原为正常格式的代码
    detokenized_code = processor.detokenize_code(line)
    
    # 构造输出文件路径
    output_path = os.path.join(output_dir, f"detokenized_{i}.tok")
    
    # 将结果写入tok文件
    with open(output_path, 'w') as file:
        file.write(detokenized_code)
    
    print(f"Detokenized code written to file: {output_path}")
