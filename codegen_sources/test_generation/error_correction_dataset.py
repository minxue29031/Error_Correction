# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import csv
import argparse
import pandas as pd
import fastBPE
from codegen_sources.model.translate import Translator
from pathlib import Path

os.makedirs('data/parallel_corpus/'+'error_correction_dataset')
def get_arguments():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--model_path", type=str, help="path to the translation model",
    )
    parser.add_argument(
        "--bpe_path", type=str, help="where the files should be outputed",
    )
    parser.add_argument(
        "--input_path", help="path to the input files",
    )
    parser.add_argument(
        "--output_path", help="path to the output files",
    )
    args=parser.parse_args()
    return args


def main(model_path,bpe_path,bpe_model,input_path,output_path):
    #translate python to java
    dict=[]
    keywords=[]
    ec_keywords1=[]
    ec_keywords2=[]

    input_path=Path(input_path)
    output_path=Path(output_path)
    beam_size=1

    df=pd.read_csv(input_path.joinpath('extract_java_function.csv'))
    transcoder = Translator(model_path, bpe_path, global_model=True)
    res = [[] for _ in range(beam_size)]
    for i, func in enumerate(df["python_function"]):
        translations = transcoder.translate(
            func,
            "python",
            "java",
            beam_size=beam_size,
            tokenized=True,
            detokenize=False,
            max_tokens=1024,
        )
        for i, res_i in enumerate(translations):
            res[i].append(res_i)

    for i, res_i in enumerate(res):
        df[f"translated_java_functions_beam_{i}"] = res_i
    df.to_csv(output_path.joinpath('pre_ec_dataset.csv'), index=False)


    #Create java error correction dataset(no correct language pair)
    csv_keyword=csv.reader(open(output_path.joinpath('pre_ec_dataset.csv'), 'r'))
    for row in csv_keyword:
        keywords.append(row)
    dict.append(keywords[0])

    for i in range(1,len(keywords)):
        if keywords[i][19].strip() != keywords[i][21].strip():
            dict.append(keywords[i])
        pd.DataFrame(dict).to_csv(output_path.joinpath('different_ec_dataset.csv'), index=False)

    #remove duplicates
    final_dataset=pd.read_csv(output_path.joinpath('different_ec_dataset.csv'), encoding='utf-8', header=1)
    final_dataset_dr=final_dataset.drop_duplicates(subset=['java_function'],keep='first',inplace=False)
    final_dataset_dr.to_csv(output_path.joinpath('final_ec_dataset.csv'), index=False)

    #Divide the dataset into training dataset and testing dataset
    train_ec=pd.read_csv(output_path.joinpath('final_ec_dataset.csv'))
    train_ec.drop(train_ec.index[int(len(train_ec)*0.6):len(train_ec)],inplace=True)
    train_ec.to_csv(output_path.joinpath('train_ec_datastore.csv'),index=False,encoding="utf-8")
    
    test_ec=pd.read_csv(output_path.joinpath('final_ec_dataset.csv'))
    test_ec.drop(test_ec.index[0:int(len(test_ec)*0.6)],inplace=True)
    test_ec.to_csv(output_path.joinpath('test_ec_datastore.csv'),index=False,encoding="utf-8")

    #Write .tok train dataset
    dataset1=open(output_path.joinpath('train_ori_java.sa.tok'),"w")
    dataset2=open(output_path.joinpath('train_ec_java.sa.tok'),"w")
    train_ec_tok=pd.read_csv(output_path.joinpath('train_ec_datastore.csv'))
    for i in range(0,len(train_ec_tok)):
        dataset1.write(train_ec_tok.iloc[i]["java_function"].strip()+"\n")
        dataset2.write(train_ec_tok.iloc[i]["translated_java_functions_beam_0"].strip()+"\n")
    dataset1.close()
    dataset2.close()

    #Write .tok test dataset
    dataset3=open(output_path.joinpath('test_ori_java.sa.tok'),"w")
    dataset4=open(output_path.joinpath('test_ec_java.sa.tok'),"w")
    test_ec_tok=pd.read_csv(output_path.joinpath('test_ec_datastore.csv'))
    for i in range(0,len(test_ec_tok)):
        dataset3.write(test_ec_tok.iloc[i]["TARGET_CLASS"].strip())
        dataset3.write(" | ")
        dataset3.write(test_ec_tok.iloc[i]["java_function"].strip()+"\n")

        dataset4.write(test_ec_tok.iloc[i]["TARGET_CLASS"].strip())
        dataset4.write(" | ")
        dataset4.write(test_ec_tok.iloc[i]["translated_java_functions_beam_0"].strip()+"\n")
    dataset3.close()
    dataset4.close()

    #Write train_ori_java to .bpe file
    with open(output_path.joinpath('train_ori_java.sa.tok'),"r") as f:
        data_ori=f.readlines()
        data_ori = bpe_model.apply([f.strip() for f in data_ori])
        dataset1_bpe = open(output_path.joinpath(f"train_ori_java.bpe"),"w",)
        for i in range(len(data_ori)):
            dataset1_bpe.write(data_ori[i].strip()+"\n")
        dataset1_bpe.close()

    #Write train_ec_java to .bpe file
    with open(output_path.joinpath('train_ec_java.sa.tok'),"r") as f:
        data_ec=f.readlines()
        data_ec = bpe_model.apply([f.strip() for f in data_ec])
        dataset2_bpe = open(output_path.joinpath(f"train_ec_java.bpe"),"w",)
        for i in range(len(data_ec)):
            dataset2_bpe.write(data_ec[i].strip()+"\n")
        dataset2_bpe.close()



if __name__ == "__main__":
    args=get_arguments()
    bpe_model = fastBPE.fastBPE(args.bpe_path)
    main(args.model_path,args.bpe_path,bpe_model,args.input_path,args.output_path)
