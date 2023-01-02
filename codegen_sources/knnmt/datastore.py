from typing import Optional
from tqdm import tqdm

from codegen_sources.model.translate import Translator
from codegen_sources.knnmt.knnmt import KNNMT
from codegen_sources.knnmt import load_functions
import numpy as np
import threading
from pycuda import driver

LANGUAGE_PAIRS = [
    "python_python",
]


class GPUThread(threading.Thread):
    def __init__(self, gpuid, knnmt, language_pair, chunk, pbar, is_validation):
        threading.Thread.__init__(self)

        self.ctx = driver.Device(gpuid).make_context()
        self.device = self.ctx.get_device()

        self.knnmt = knnmt
        self.language_pair = language_pair
        self.chunk = chunk
        self.pbar = pbar
        self.is_validation = is_validation

    def run(self):
        print("Add to dataset", self.getName(), self.device.name(), self.ctx.get_api_version())

        src_language = self.language_pair.split("_")[0]
        tgt_language = self.language_pair.split("_")[1]

        translator_path = f"models/TransCoder_model_2.pth"
        translator = Translator(translator_path, "data/bpe/cpp-java-python/codes", global_model=True)

        # Obtain features and targets from decoder
        for src_sample, tgt_sample in self.chunk:
            decoder_features, _, targets, target_tokens, input_code, output_code = translator.get_features(
                input_code=src_sample,
                target_code=tgt_sample,
                src_language=src_language,
                tgt_language=tgt_language,
                tokenized=not self.is_validation
            )

            self.knnmt.add_to_datastore(decoder_features, targets, input_code, output_code, self.language_pair)
            self.pbar.update(1)

    def join(self):
        self.ctx.detach()
        threading.Thread.join(self)


def add_to_datastore(knnmt: KNNMT, parallel_functions, is_validation: bool=False):
    driver.init()

    for language_pair in parallel_functions.keys():
        print("#" * 10 + f" Creating Datastore for '{language_pair}' " + "#" * 10)

        # Get parallel functions for language pair and split into chunks
        functions = parallel_functions[language_pair]
        chunks = np.array_split(functions, driver.Device.count())

        # Add functions to kNN-MT datastore
        with tqdm(total=len(functions)) as pbar:
            threads = [
                GPUThread(index, knnmt, language_pair, chunk, pbar, is_validation)
                for index, chunk in enumerate(chunks)
            ]

            for thread in threads:
                thread.start()

            for thread in threads:
                thread.join()

        knnmt.save_datastore(language_pair)


def train_datastore(knnmt: KNNMT, language_pair: Optional[str]):
    if language_pair is not None:
        knnmt.train_datastore(language_pair)
        return

    for language_pair in LANGUAGE_PAIRS:
        knnmt.train_datastore(language_pair)


# Create kNN-MT datastore from parallel corpus_1
knnmt_parallel_corpus = KNNMT("out/knnmt/parallel_corpus1")
parallel_functions = load_functions.load_parallel_functions("data/parallel_corpus/error_correction_dataset")
add_to_datastore(knnmt_parallel_corpus, parallel_functions)
train_datastore(knnmt_parallel_corpus,"python_python")

# Create kNN-MT datastore from parallel corpus_2
knnmt_parallel_corpus = KNNMT("out/knnmt/parallel_corpus2")
parallel_functions = load_functions.load_parallel_functions("data/parallel_corpus/error_correction_dataset")
add_to_datastore(knnmt_parallel_corpus, parallel_functions)
train_datastore(knnmt_parallel_corpus,"python_python")

#mix
load_functions.created_mixed_datastore("out/knnmt")
knnmt_mixed = KNNMT("out/knnmt/mixed")
train_datastore(knnmt_mixed,"python_python")


