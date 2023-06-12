import numpy as np

def extract_values(keys_file, inputs_file, values_file):
    # Read values from values_python_python.npy file
    values = np.load(values_file)
    
    # Read keys and inputs from keys_python_python.npy and inputs_python_python.npy files
    keys = np.load(keys_file)
    inputs = np.load(inputs_file)

    # Create lists to store the extracted values
    extracted_values = []
    extracted_keys = []
    extracted_inputs = []

    # Iterate over each row in values
    for i, row in enumerate(values):
        if row == 1:
            # If the value is equal to 1, extract the corresponding content
            extracted_values.append(row)
            extracted_keys.append(keys[i])
            extracted_inputs.append(inputs[i])

    # Save the extracted content as new .npy files
    np.save('extracted_values.npy', np.array(extracted_values))
    np.save('extracted_keys.npy', np.array(extracted_keys))
    np.save('extracted_inputs.npy', np.array(extracted_inputs))

    # Return the shape of the extracted content
    return np.array(extracted_values).shape, np.array(extracted_keys).shape, np.array(extracted_inputs).shape


def main():
    keys_file = 'keys_python_python.npy'
    inputs_file = 'inputs_python_python.npy'
    values_file = 'values_python_python.npy'

    extracted_values_shape, extracted_keys_shape, extracted_inputs_shape = extract_values(keys_file, inputs_file, values_file)

    print("extracted_values_shape:", extracted_values_shape)
    print("extracted_keys_shape:", extracted_keys_shape)
    print("extracted_inputs_shape:", extracted_inputs_shape)


if __name__ == '__main__':
    main()
