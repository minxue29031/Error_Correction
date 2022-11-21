EXTRACT_DATASET='data/parallel_corpus/extract'
INPUT_ALL='data/parallel_corpus/results/transcoder_outputs/python_transcoder_translation.csv'
INPUT_JAVA_CONTENT='data/parallel_corpus/offline_dataset/train.java_sa-python_sa.java_sa.tok' 

python -m codegen_sources.test_generation.extract_java_BT\
    --output_path $EXTRACT_DATASET\
    --csv_path $INPUT_ALL\
    --extract_target $INPUT_JAVA_CONTENT\