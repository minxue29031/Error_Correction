MODEL_PATH='models/TransCoder_model_2.pth'
BPE_PATH='data/bpe/cpp-java-python/codes'
INPUT_DATASET='data/test_dataset'
OUTPUT_EC_PATH='data/parallel_corpus_test'

python -m codegen_sources.test_generation.dataset_add\
    --model_path $MODEL_PATH\
    --bpe_path $BPE_PATH\
    --pre_data_path $INPUT_DATASET\
    --output_path $OUTPUT_EC_PATH\

