INPUT_DATA='data/parallel_corpus/extract/ori_java.sa.tok'
MODEL_PATH='models/TransCoder_model_2.pth'
BPE_PATH='data/bpe/cpp-java-python/codes'
INPUT_DATASET='data/parallel_corpus/extract'
OUTPUT_EC_PATH='data/parallel_corpus/error_correction_dataset'

python -m codegen_sources.test_generation.error_correction_dataset\
    --input_data $INPUT_DATA\
    --model_path $MODEL_PATH\
    --bpe_path $BPE_PATH\
    --input_path $INPUT_DATASET\
    --output_path $OUTPUT_EC_PATH\
