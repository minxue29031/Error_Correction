import os
import subprocess
import numpy as np
from codegen_sources.model.translate import Translator
from codegen_sources.knnmt.load_functions import extract_functions
from codegen_sources.knnmt.knnmt import KNNMT
#os.makedirs('data/'+'ec_test_datastore')
DATASET_PATH = "data/parallel_corpus/error_correction_dataset"

def error_correction_output(knnmt: KNNMT, translator: Translator, language_pair: str):
    src_language, tgt_language = language_pair.split("_")[0], language_pair.split("_")[1]

    # Get tokenized source function
    source_functions = extract_functions(f"{DATASET_PATH}/test_ec_java.sa.tok")

    for i in range(len(source_functions)):
        source = source_functions[i]
        source = translator.tokenize(source, src_language)
        generated = ""
        inputs = ""
        tc_prediction = ""

        # Get KNN-MT translation
        translator.use_knn_store = True
        translation = translator.translate(source, src_language, tgt_language)[0]
        
        data_ec=open(f"{DATASET_PATH}/test_ec_java.sa.tok","r").readlines()        
        writefile = open("data/ec_test_datastore/"+data_ec[i].split(" | ")[0]+".java","w")        
        writefile.write(
        """
import java.util.*;
import java.util.stream.*;
import java.lang.*;
import javafx.util.Pair;\n
"""
        )
        writefile.write("public class " + data_ec[i].split(" | ")[0] + "{\n")
        writefile.write(translation)
        writefile.write("}\n")
        writefile.close()


        # Get original TransCoder translation
        translator.use_knn_store = False
        original_translation = translator.translate(source, src_language, tgt_language)[0]

        print("-----------------original-TransCoder-translation-------------")
        print("\n")
        print(original_translation)
        print("\n")

        print("-----------------KNN-MT-translation-------------")
        print("\n")
        print(translation)

if __name__ == "__main__":
    #KNN-MT parameters
    knnmt = KNNMT("out/knnmt/parallel_corpus")
    knnmt_k=8
    knnmt_lambda=0.3
    knnmt_temperature=10
    knnmt_tc_temperature=5

    knnmt_params = {
        "k": knnmt_k,
        "lambda": knnmt_lambda,
        "temperature": knnmt_temperature,
        "tc_temperature": knnmt_tc_temperature
    }


    #TransCoder model parameters
    language_pair = "java_java"
    src_language = language_pair.split("_")[0]
    tgt_language = language_pair.split("_")[1]
    translator_path = f"models/TransCoder_model_1.pth"

    translator = Translator(
        translator_path,
        "data/bpe/cpp-java-python/codes",
        global_model=True,
        knnmt_dir=knnmt.knnmt_dir,
        knnmt_params=knnmt_params,
    )

    #Error_Correction model output
    error_correction_output(knnmt, translator, language_pair)