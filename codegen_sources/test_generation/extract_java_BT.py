#!python /content/drive/MyDrive/Error_Correction/codegen_sources/test_generation/extract_java_BT.py 
#--output_path 'data/parallel_corpus/extract' 
#--csv_path 'data/parallel_corpus/results/transcoder_outputs/python_transcoder_translation.csv' 
#--extract_target 'data/parallel_corpus/offline_dataset/train.java_sa-python_sa.java_sa.tok'


import sys
import csv
import argparse
import pandas as pd
import os
from pathlib import Path

os.makedirs('data/parallel_corpus/'+'extract')

def get_arguments():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--output_path", help="path to the output files",
    )
    parser.add_argument(
        "--csv_path", type=str, help="where the csv file should be inputed",
    )
    parser.add_argument(
        "--extract_target", type=str, help="where the target file should be inputed",
    )
    parser.add_argument(
        "--translated_func", type=str, help="where the python/cpp file should be inputed",
    )
    args = parser.parse_args()
    return args

def main(output_path,csv_path,extract_target,translated_func):
    result=[]
    translated_result=[]
    dict1=[]
    dict2=[]
    dict=[]
    keywords=[]
    csv_path=Path(csv_path)
    extract_target=Path(extract_target)
    output_path=Path(output_path)
    translated_func=Path(translated_func)

    csv_keyword=csv.reader(open(csv_path, 'r'))
    with open(extract_target,'r') as f:
      for line in f:
        result.append(list(line.strip('\n').split('\n')))

    with open(translated_func,'r') as f2:
      for line in f2:
        translated_result.append(list(line.strip('\n').split('\n')))

    for row in csv_keyword:
        keywords.append(row)
    dict1.append(keywords[0])

    dict2.append("python_function")
    for i in range(len(keywords)):
        for j in range(len(result)):
            if result[j][0] in keywords[i][19]:
                print(keywords[i])
                dict1.append(keywords[i])
                dict2.append(translated_result[j][0])
            dir_all=pd.DataFrame(dict1)
            dir_python = pd.DataFrame(dict2)
            df=pd.concat([dir_all,dir_python],axis=1).to_csv(output_path.joinpath(f"write_java_function.csv"), index=False)

    write_function=pd.read_csv(output_path.joinpath(f"write_java_function.csv"), encoding='utf-8', header=1)
    write_function.to_csv(output_path.joinpath(f'pre_extract_java.csv'), index=False)

    pre_function=pd.read_csv(output_path.joinpath(f'pre_extract_java.csv'))
    function=pre_function.drop(['translated_python_functions_beam_0'],axis=1)
    function.to_csv(output_path.joinpath(f'extract_java_function.csv'), index=False)


if __name__ == "__main__":
    args = get_arguments()
    main(args.output_path,args.csv_path,args.extract_target,args.translated_func)

