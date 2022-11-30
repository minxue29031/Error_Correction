# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


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
    args = parser.parse_args()
    return args


def main(output_path,csv_path,extract_target):
    result=[]
    dict=[]
    keywords=[]
    csv_path=Path(csv_path)
    extract_target=Path(extract_target)
    output_path=Path(output_path)

    csv_keyword=csv.reader(open(csv_path, 'r'))
    with open(extract_target,'r') as f:
      for line in f:
        result.append(list(line.strip('\n').split('\n')))

    for row in csv_keyword:
        keywords.append(row)
    dict.append(keywords[0])

    for i in range(len(keywords)):
        for j in range(len(result)):
            if result[j][0] in keywords[i][19]:
                print(keywords[i])
                dict.append(keywords[i])
            pd.DataFrame(dict).to_csv(output_path.joinpath(f"write_java_function.csv"), index=False)


    write_function=pd.read_csv(output_path.joinpath(f"write_java_function.csv"), encoding='utf-8', header=1)
    write_function.to_csv(output_path.joinpath(f'pre_extract_java.csv'), index=False)

    pre_function=pd.read_csv(output_path.joinpath(f'pre_extract_java.csv'))
    function=pre_function.drop(['translated_python_functions_beam_0'],axis=1)
    function.to_csv(output_path.joinpath(f'extract_java_function.csv'), index=False)

    #Write .tok dataset
    dataset_trans_pre=open(output_path.joinpath('ori_java.sa.tok'),"w")
    read_ori_java=pd.read_csv(output_path.joinpath('extract_java_function.csv'))
    for i in range(0,len(read_ori_java)):
        dataset_trans_pre.write(read_ori_java.iloc[i]["java_function"].strip()+"\n")
    dataset_trans_pre.close()


if __name__ == "__main__":
    args = get_arguments()
    main(args.output_path,args.csv_path,args.extract_target)

