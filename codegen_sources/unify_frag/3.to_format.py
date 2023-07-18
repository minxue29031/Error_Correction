from codegen_sources.preprocessing.lang_processors.python_processor import PythonProcessor
import os

input_dir = 'data/ec_result_tcst/unify/repla_func_var_para/'
output_dir = 'data/ec_result_tcst/unify/unify_result/'
output_file = 'data/ec_result_tcst/unify/test_unify.tok'

processor = PythonProcessor()

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Iterate over the repla_func_var_para files
for i in range(8069):
    input_file = f'{input_dir}repla_func_var_para_{i}.tok'
    output_file_i = f'{output_dir}test_unify_{i}.tok'

    with open(input_file, 'r') as f:
        input_code = f.read()

    tokenized_code = processor.tokenize_code(input_code)

    with open(output_file_i, 'w') as f:
        f.write(' '.join(tokenized_code))

    print(f"Tokenization completed for {input_file}. Output saved in {output_file_i}.")



# Concatenate the generated files into a single output file
with open(output_file, 'w') as f:
    for i in range(8069):
        output_file_i = f'{output_dir}test_unify_{i}.tok'
        with open(output_file_i, 'r') as f_i:
            content = f_i.read()
            f.write(content.strip() + '\n')  # 写入内容并换行

print(f"All files merged into {output_file}.")