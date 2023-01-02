from hashlib import sha256
from typing import Optional
import numpy as np
import os

os.makedirs('out/knnmt/mixed/'+'datastore')
LANGUAGE_PAIRS = ["python_python",]

def created_mixed_datastore(knnmt_dir: str):
    for language_pair in LANGUAGE_PAIRS:
        # Load datastore keys from the parallel corpus and from the validation set
        datastore_keys = np.load(f"{knnmt_dir}/parallel_corpus1/datastore/keys_{language_pair}.npy")
        datastore_keys_valid = np.load(f"{knnmt_dir}/parallel_corpus2/datastore/keys_{language_pair}.npy")

        # Load datastore values from the parallel corpus and from the validation set
        datastore_values = np.load(f"{knnmt_dir}/parallel_corpus1/datastore/values_{language_pair}.npy")
        datastore_values_valid = np.load(f"{knnmt_dir}/parallel_corpus2/datastore/values_{language_pair}.npy")

        # Load datastore inputs from the parallel corpus and from the validation set
        datastore_inputs = np.load(f"{knnmt_dir}/parallel_corpus1/datastore/inputs_{language_pair}.npy")
        datastore_inputs_valid = np.load(f"{knnmt_dir}/parallel_corpus2/datastore/inputs_{language_pair}.npy")

        # Concatenate datastores
        datastore_keys = np.concatenate((datastore_keys, datastore_keys_valid), axis=0)
        datastore_values = np.concatenate((datastore_values, datastore_values_valid), axis=0)
        datastore_inputs = np.concatenate((datastore_inputs, datastore_inputs_valid), axis=0)

        print("Keys", datastore_keys.shape)
        print("Values", datastore_values.shape)
        print("Inputs", datastore_inputs.shape)

        # Save datastores
        np.save(f"{knnmt_dir}/mixed/datastore/keys_{language_pair}.npy", datastore_keys)
        np.save(f"{knnmt_dir}/mixed/datastore/values_{language_pair}.npy", datastore_values)
        np.save(f"{knnmt_dir}/mixed/datastore/inputs_{language_pair}.npy", datastore_inputs)



def load_parallel_functions(dataset_path: str, language_pair: str = None):
    parallel_functions = {}

    # Load parallel functions for language pairs containing Java and Python
    if language_pair is None or language_pair == "java_java":
        java_python_java_ori = open(f"{dataset_path}/train_ori_java.bpe", "r")
        java_ori_functions = java_python_java_ori.readlines()

        java_python_java_ec = open(f"{dataset_path}/train_ec_java.bpe", "r")
        java_ec_functions = java_python_java_ec.readlines()

        #parallel_functions["java_java"] = list(zip(java_ori_functions, java_ec_functions))
        parallel_functions["java_java"] = list(zip(java_ec_functions, java_ori_functions))

    # Deduplicate parallel functions
    parallel_functions = deduped_parallel_functions(parallel_functions)

    if language_pair is not None:
        return parallel_functions[language_pair]

    return parallel_functions

def load_validation_functions(validation_set_path: str, language_pair: str = None, half: Optional[int] = None):
    # Load Java functions
    if language_pair is None or "java" in language_pair:
        java_functions = extract_functions(f"{validation_set_path}/transcoder_valid.java.tok")

    parallel_functions = {}

    # Combine functions to obtain parallel functions
    if language_pair is None or language_pair == "java_java":
        parallel_functions["cpp_java"] = list(zip(java_functions, java_functions))


    if language_pair is not None:
        if half is None:
            # Return parallel function for language pair
            return parallel_functions[language_pair]
        elif half == 1:
            # Return first half of parallel functions
            return parallel_functions[language_pair][:int(len(parallel_functions[language_pair]) / 2)]
        else:
            # Return second half of parallel functions
            return parallel_functions[language_pair][int(len(parallel_functions[language_pair]) / 2):]

    return parallel_functions



def extract_functions(path):
    def extract_function(line):
        return " | ".join(line.split(" | ")[1:])

    file = open(path, "r")
    return [extract_function(line) for line in file.readlines()]


def deduped_parallel_functions(parallel_functions):
    deduped_functions = {}

    for language_pair in parallel_functions.keys():
        # Get parallel functions of language pair
        function_pairs = parallel_functions[language_pair]
        deduped_pairs = []
        hashes = {}

        for pair in function_pairs:
            source, target = pair

            # Skip empty lines
            if source == "\n" or target == "\n":
                continue

            # Hash source and target function
            hash = sha256(source.encode("utf8") + target.encode("utf8")).hexdigest()

            # If hash has not been generated before, add to deduplicated list
            if hashes.get(hash) is None:
                hashes[hash] = pair
                deduped_pairs.append(pair)

        deduped_functions[language_pair] = deduped_pairs

        print(f"Size of '{language_pair}' dataset: {len(function_pairs)}")
        print(f"Size of deduped '{language_pair}' dataset: {len(deduped_pairs)}")

    return deduped_functions
