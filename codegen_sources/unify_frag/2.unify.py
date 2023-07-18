import ast
import builtins
import os

# build folder
os.makedirs('data/ec_result_tcst/unify/repla_func', exist_ok=True)
os.makedirs('data/ec_result_tcst/unify/repla_func_var', exist_ok=True)
os.makedirs('data/ec_result_tcst/unify/repla_func_var_para', exist_ok=True)

def replace_function_names(input_file, output_file):
    # Read file content
    with open(input_file, 'r') as file:
        content = file.read()

    # Parse code into AST
    tree = ast.parse(content)

    # Replace function names with 'func'
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            node.name = 'func'

    # Convert AST back to code
    modified_code = ast.unparse(tree)

    # Write the result to a file
    with open(output_file, 'w') as file:
        file.write(modified_code)

    print("Modified code with replaced function names written to file:", output_file)


def replace_variable_names(input_file, output_file):
    with open(input_file, 'r') as file:
        code = file.read()

    # Parse code into AST
    tree = ast.parse(code)

    # Extract function definitions
    functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

    # Traverse each function definition
    for func in functions:
        # Extract variable names within the function
        variable_names = []
        for node in ast.walk(func):
            if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Load)):
                if node.id[0].isalpha() and not is_builtin_name(node.id) and not is_module_name(node.id) and not is_parameter_name(node, func):
                    variable_names.append(node.id)

        # Build replacement mapping for variable names
        variable_mapping = {}
        for name in variable_names:
            if name not in variable_mapping:
                index = variable_names.index(name)
                new_name = chr(ord('b') + index % 24)
                variable_mapping[name] = new_name

        # Replace variable names within the function
        for node in ast.walk(func):
            if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Load)) and node.id in variable_mapping:
                node.id = variable_mapping[node.id]

    # Convert the modified AST back to code
    modified_code = ast.unparse(tree)

    # Write the result to a file
    with open(output_file, 'w') as file:
        file.write(modified_code)


def replace_parameter_names(input_file, output_file):
    # Read file content
    with open(input_file, 'r') as file:
        content = file.read()

    # Parse code into AST
    tree = ast.parse(content)

    # Generate replacement parameter names
    replacement_names = ['a{}'.format(i) for i in range(50)]

    # Record the mapping of parameter names
    parameter_mapping = {}

    # Replace parameter names
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for i, arg in enumerate(node.args.args):
                old_name = arg.arg
                new_name = replacement_names[i]

                # Update parameter name mapping
                parameter_mapping[old_name] = new_name

                # Replace parameter name
                arg.arg = new_name

                # Replace parameter names within the function body
                for func_node in ast.walk(node):
                    if isinstance(func_node, ast.Name) and func_node.id == old_name:
                        func_node.id = new_name

    # Write the modified code to a file
    with open(output_file, 'w') as file:
        file.write(ast.unparse(tree))

    # Print the parameter name mapping
    for old_name, new_name in parameter_mapping.items():
        print(f'{old_name} -> {new_name}')

    print(f"Modified python code written to file: {output_file}")


# Utility functions

# Check if a variable name is a built-in name
def is_builtin_name(name):
    return name in dir(builtins)


# Check if a variable name is a module name
def is_module_name(name):
    try:
        __import__(name)
        return True
    except ImportError:
        return False


# Check if a variable name is a parameter name of a function
def is_parameter_name(node, func):
    if isinstance(func.args, ast.arguments):
        for arg in func.args.args:
            if arg.arg == node.id:
                return True
    return False


# Usage
for i in range(8069):
    input_file = f'data/ec_result_tcst/unify/detokenized/detokenized_{i}.tok'
    output_file1 = f'data/ec_result_tcst/unify/repla_func/repla_func_{i}.tok'
    output_file2 = f'data/ec_result_tcst/unify/repla_func_var/repla_func_var_{i}.tok'
    output_file3 = f'data/ec_result_tcst/unify/repla_func_var_para/repla_func_var_para_{i}.tok'
    replace_function_names(input_file, output_file1)
    replace_variable_names(output_file1, output_file2)
    replace_parameter_names(output_file2, output_file3)
