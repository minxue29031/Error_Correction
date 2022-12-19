import os
import re
import subprocess
import numpy as np
from codegen_sources.model.translate import Translator
from codegen_sources.knnmt.load_functions import extract_functions
from codegen_sources.knnmt.knnmt import KNNMT
from codegen_sources.preprocessing.lang_processors.python_processor import (
    PythonProcessor,
)

processor = PythonProcessor()

#os.makedirs('data/'+'ec_result_py')

def error_correction_output(knnmt: KNNMT, translator: Translator, language_pair: str):
    src_language, tgt_language = language_pair.split("_")[0], language_pair.split("_")[1]

    #Get tokenized source function
    source_functions = open("data/ec_result_py/test_input/test_dataset.sa.tok").readlines()
    print("***source_functions***:",source_functions)

    dataset_res=[]
    out = open("data/ec_result_py/write_ec_results.sa.tok","w") 
    out_noknn = open("data/ec_result_py/write_no_ec_results.sa.tok","w") 
    for i in range(len(source_functions)):
        source = source_functions[i]
        source = translator.tokenize(source, src_language)
        generated = ""
        inputs = ""
        tc_prediction = ""

        # Get KNN-MT translation
        print("---------------------KNN-MT-translation----------------------")
        translator.use_knn_store = True
        translation = translator.translate(source, src_language, tgt_language)[0]
        print(translation)

        translation_tok=translator.tokenize(translation, src_language)
        out.write(translation_tok)
        out.write('\n')

        # Get original TransCoder translation
        print("-----------------original-TransCoder-translation-------------")
        translator.use_knn_store = False
        original_translation = translator.translate(source, src_language, tgt_language)[0]
        print(original_translation)

        ori_translation_tok=translator.tokenize(original_translation, src_language)
        out.write(ori_translation_tok)
        out.write('\n')

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
    language_pair = "python_python"
    src_language = language_pair.split("_")[0]
    tgt_language = language_pair.split("_")[1]
    translator_path = f"models/TransCoder_model_2.pth"

    translator = Translator(
        translator_path,
        "data/bpe/cpp-java-python/codes",
        global_model=True,
        knnmt_dir=knnmt.knnmt_dir,
        knnmt_params=knnmt_params,
    )

    #Error_Correction model output
    error_correction_output(knnmt, translator, language_pair)