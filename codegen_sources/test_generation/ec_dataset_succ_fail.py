# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#!python codegen_sources/test_generation/ec_dataset_succ_fail.py --input_path 'data/parallel_test' --output_path 'data/parallel_test/java_language_pair'

import csv 
import os
import numpy as np
import pandas as pd
import argparse
from pathlib import Path

#os.makedirs('data/parallel_test/'+'java_language_pair')
def get_arguments():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--input_path", help="path to the input files",
    )
    parser.add_argument(
        "--output_path", help="path to the output files",
    )
    args = parser.parse_args()
    return args

def main(input_path,output_path):
    dict=[]
    keywords=[]
    success=[]
    failure=[]
    beam_size=4
    t=beam_size+1

    output_path=Path(output_path)
    input_path=Path(input_path)
    extract_function(input_path)

    #Write the successful and failure java functions to sa.tok respectively
    csv_keyword=csv.reader(open(input_path.joinpath(f"extract_func.csv"), 'r'))
    success_data=open(input_path.joinpath(f"success_java.tok"),"w")
    failure_data=open(input_path.joinpath(f"failure_java.tok"),"w")

    for row in csv_keyword:
        keywords.append(row)

    for i in range(len(keywords)):
        for j in range(len(keywords[i])):
            if "('success', " in keywords[i][j]:
                success_data.write(keywords[i][j-t].strip()+",,,") 
                
            if "('failure', " in keywords[i][j]:
                failure_data.write(keywords[i][j-t].strip()+",,,")

        failure_data.write("\n")
        success_data.write("\n")

    failure_data.close()
    success_data.close()

    #Extract the best successful java_function
    success_formal=open(input_path.joinpath(f"success_formal.tok"),"w")
    with open(input_path.joinpath(f"success_java.tok"),'r') as f:
        for line in f:
            success.append(list(line.strip('\n').split(',,,')))

        for i in range(len(success)):
            success_formal.write(success[i][0].strip()+"\n")
    success_formal.close()

    #Generate java language pair
    df_1= open(input_path.joinpath(f"success_formal.tok"),'r')
    df_2= open(input_path.joinpath(f"failure_java.tok"),'r')
    for line in df_2:
        failure.append(list(line.strip('\n').split(',,,')))

    java_pair_success= open(output_path.joinpath(f"java_pair_success.tok"),'w')
    java_pair_failure= open(output_path.joinpath(f"java_pair_failure.tok"),'w')
    for i in range(len(failure)):
        for j in range(len(failure[i])-1):
            java_pair_success.write(success[i][0].strip()+"\n")
            java_pair_failure.write(failure[i][j].strip()+"\n")

    java_pair_success.close()
    java_pair_failure.close()


def extract_function(input_path):
    dict=[]
    keywords=[]
    df=open(input_path.joinpath(f"test_results_python.csv"))
    reader=csv.reader(df)

    for row in reader:
        for subrow in row:
            if "TARGET_CLASS" in subrow:
                dict.append(row)

            if "('success'" in subrow:
                dict.append(row)
    dir = pd.DataFrame(dict).to_csv(input_path.joinpath(f"extract_func_pr.csv"), index=False)

    extract_func=pd.read_csv(input_path.joinpath(f"extract_func_pr.csv"), encoding='utf-8', header=1).drop_duplicates()
    extract_func.to_csv(input_path.joinpath(f"extract_func.csv"), index=False)


if __name__ == "__main__":
    args = get_arguments()
    main(args.input_path,args.output_path)
