import numpy as np

def process_npy(value_file, keys_file, inputs_file):
    values_result = []
    keys_result = []
    inputs_result = []

    values_data = np.load(value_file)
    keys_data = np.load(keys_file)
    inputs_data = np.load(inputs_file)

    prev_value = None
    for i, value in enumerate(values_data):
        if value == 1:
            if prev_value is not None:
                values_result.append(prev_value)
                keys_result.append(keys_data[i-1])
                inputs_result.append(inputs_data[i-1])
            values_result.append(value)
            keys_result.append(keys_data[i])
            inputs_result.append(inputs_data[i])
        prev_value = value

    return values_result, keys_result, inputs_result


def remove_rows(values, keys, inputs):
    values_result = []
    keys_result = []
    inputs_result = []

    for i, value in enumerate(values):
        if value != 1:
            values_result.append(value)
            keys_result.append(keys[i])
            inputs_result.append(inputs[i])

    return values_result, keys_result, inputs_result


def main():
    # Process the values_python_python.npy file and retrieve the results
    value_file = 'values_python_python.npy'
    keys_file = 'keys_python_python.npy'
    inputs_file = 'inputs_python_python.npy'
    values_result, keys_result, inputs_result = process_npy(value_file, keys_file, inputs_file)

    # Save the results as new NPY files
    values_output_file = 'values_python_python_new.npy'
    keys_output_file = 'keys_python_python_new.npy'
    inputs_output_file = 'inputs_python_python_new.npy'
    np.save(values_output_file, np.array(values_result))
    np.save(keys_output_file, np.array(keys_result))
    np.save(inputs_output_file, np.array(inputs_result))

    # Remove rows with token=1 from the values_python_python_new.npy file, and delete the corresponding rows from keys_python_python_new.npy and inputs_python_python_new.npy
    values_final, keys_final, inputs_final = remove_rows(values_result, keys_result, inputs_result)

    # Save as new NPY files
    values_final_output_file = 'values_python_python_final.npy'
    keys_final_output_file = 'keys_python_python_final.npy'
    inputs_final_output_file = 'inputs_python_python_final.npy'

    np.save(values_final_output_file, np.array(values_final))
    np.save(keys_final_output_file, np.array(keys_final))
    np.save(inputs_final_output_file, np.array(inputs_final))

    # Output the results
    print("values_python_python_new.npy content:", np.load(values_output_file).shape)
    print("keys_python_python_new.npy content:", np.load(keys_output_file).shape)
    print("inputs_python_python_new.npy content:", np.load(inputs_output_file).shape)

    print("values_python_python_final.npy content:", np.load(values_final_output_file).shape)
    print("keys_python_python_final.npy content:", np.load(keys_final_output_file).shape)
    print("inputs_python_python_final.npy content:", np.load(inputs_final_output_file).shape)


if __name__ == '__main__':
    main()
