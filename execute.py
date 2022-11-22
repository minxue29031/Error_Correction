import os
import subprocess
import numpy as np
from codegen_sources.model.translate import Translator
from codegen_sources.knnmt.load_functions import extract_functions
from codegen_sources.knnmt.knnmt import KNNMT
from codegen_sources.knnmt.datastore import add_to_datastore, train_datastore

DATASET_PATH = "data/test_dataset"

def output_sample(knnmt: KNNMT, translator: Translator, language_pair: str):
    src_language, tgt_language = language_pair.split("_")[0], language_pair.split("_")[1]

    # Get tokenized source function
    source_functions = extract_functions(f"{DATASET_PATH}/transcoder_test.{src_language}.tok")
    
    for i in range(len(source_functions)):
   
        source = source_functions[i]
        source = translator.tokenize(source, src_language)
        generated = ""
        inputs = ""
        tc_prediction = ""

        # Predict target tokens using kNN-MT only
        while len(generated.split(" ")) < 1000:
            tc_target, prediction, input = predict_next_token(knnmt, translator, src_language, tgt_language, source, generated)
            generated += " " + prediction
            inputs += f"{input[0]}\n{input[1]}\n"
            tc_prediction += " " + tc_target

        # Get original TransCoder translation
        translation = translator.translate(source, src_language, tgt_language)[0]
        translator.use_knn_store = False
        original_translation = translator.translate(source, src_language, tgt_language)[0]
    
        target_functions = extract_functions(f"{DATASET_PATH}/transcoder_test.{tgt_language}.tok")
        target = target_functions[i]
        target = translator.tokenize(target, tgt_language)
    
        print("\n\n\n\n\n")
        print(f"TC PREDICTION: '{tc_prediction[1:]}'")
        print(f"FINAL PREDICTION: '{generated[1:]}'")
        print(f"GROUND TRUTH: '{target} '")
        print("\n\n")

def predict_next_token(
    knnmt: KNNMT,
    translator: Translator,
    src_language: str,
    tgt_language: str,
    source: str,
    generated: str
):
    # Get hidden feature representation of last decoder layer and ground truth target tokens
    decoder_features, _, targets, target_tokens, _, _ = translator.get_features(
        input_code=source,
        target_code=generated,
        src_language=src_language,
        tgt_language=tgt_language,
        predict_single_token=True,
        tokenized=True
    )

    # Retrieve k nearest neighbors including their distances and inputs
    language_pair = f"{src_language}_{tgt_language}"
    features = decoder_features[-1].unsqueeze(0)
    knns, distances, inputs = knnmt.get_k_nearest_neighbors(features, language_pair, with_inputs=True)
    tokens = [translator.get_token(id) for id in knns[0]]
    
    # import pdb; pdb.set_trace()
    return target_tokens[-1], tokens[0], inputs[0][0]

knnmt = KNNMT("out/knnmt/parallel_corpus")
language_pair = "java_java"
src_language = language_pair.split("_")[0]
tgt_language = language_pair.split("_")[1]

translator_path = f"models/TransCoder_model_1.pth"
translator = Translator(
    translator_path,
    "data/bpe/cpp-java-python/codes",
    global_model=True,
    knnmt_dir=knnmt.knnmt_dir
)

output_sample(knnmt, translator, language_pair)