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
        "--input_data", help="path to the input files",
    )
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


def main(input_data,model_path,bpe_path,bpe_model,input_path,output_path):
    #translate python to java
    dict=[]
    keywords=[]
    input_data=Path(input_data)
    input_path=Path(input_path)
    output_path=Path(output_path)

    java_functions=[ func 
              for func in open(input_data).readlines() 
              ]

    fh=open(input_path.joinpath('translate_python_java.sa.tok'),"w")
    transcoder=Translator(model_path, bpe_path, global_model=True)
    for func in java_functions:
        translations=transcoder.translate(
            func,
            "python",
            "java",
            beam_size=1,
            tokenized=True,
            detokenize=False,
            max_tokens=1024,
            )
        fh.write(translations[0].replace("@","")+"\n")
    fh.close()

    #Convert sa.tok to csv
    frame=pd.read_table(input_path.joinpath('translate_python_java.sa.tok'), delimiter='|', header=None, error_bad_lines=False)
    frame.columns=["translated_java_functions_beam_0"]
    frame.to_csv(input_path.joinpath('translated_java.csv'), index=False)

    #Create java error correction dataset(contains the correct language pair)
    f1=pd.read_csv(input_path.joinpath('extract_java_function.csv'))
    f2=pd.read_csv(input_path.joinpath('translated_java.csv'))
    file=[f1,f2]
    comb=pd.concat(file,axis=1)
    comb.to_csv(output_path.joinpath('pre_ec_dataset.csv'), index=0, sep=',')


    #Create java error correction dataset(no correct language pair)
    csv_keyword=csv.reader(open(output_path.joinpath('pre_ec_dataset.csv'), 'r'))

    for row in csv_keyword:
        keywords.append(row)
    dict.append(keywords[0])

    dataset1=open(output_path.joinpath('ori_java.sa.tok'),"w")
    dataset2=open(output_path.joinpath('ec_java.sa.tok'),"w")

    for i in range(1,len(keywords)):
        if keywords[i][19].strip() != keywords[i][20].strip():
            dataset1.write(keywords[i][19].strip()+"\n")
            dataset2.write(keywords[i][20].strip()+"\n")
            dict.append(keywords[i])
        pd.DataFrame(dict).to_csv(output_path.joinpath('different_ec_dataset.csv'), index=False)
    dataset1.close()
    dataset2.close()

    final_dataset=pd.read_csv(output_path.joinpath('different_ec_dataset.csv'), encoding='utf-8', header=1)
    final_dataset.to_csv(output_path.joinpath('final_ec_dataset.csv'), index=False)

    #Write ori_java to .bpe file
    with open(output_path.joinpath('ori_java.sa.tok'),"r") as f:
        data_ori=f.readlines()
        data_ori = bpe_model.apply([f.strip() for f in data_ori])
        dataset1_bpe = open(output_path.joinpath(f"ori_java.bpe"),"w",)
        for i in range(len(data_ori)):
            dataset1_bpe.write(data_ori[i].strip()+"\n")
        dataset1_bpe.close()

    #Write ec_java to .bpe file
    with open(output_path.joinpath('ec_java.sa.tok'),"r") as f:
        data_ec=f.readlines()
        data_ec = bpe_model.apply([f.strip() for f in data_ec])
        dataset2_bpe = open(output_path.joinpath(f"ec_java.bpe"),"w",)
        for i in range(len(data_ec)):
            dataset2_bpe.write(data_ec[i].strip()+"\n")
        dataset2_bpe.close()

if __name__ == "__main__":
    args=get_arguments()
    bpe_model = fastBPE.fastBPE(args.bpe_path)
    main(args.input_data,args.model_path,args.bpe_path,bpe_model,args.input_path,args.output_path)


