#  Interpretable Error Correction for Code-to-Code Translation

This repository implements an **interpretable error correction method** designed to enhance **code-to-code translation**. It provides a way to correct wrong translations produced by Transformer-based models like **TransCoder-ST**, while offering interpretability for the corrections.



##  Installation

Install dependencies:

```bash
pip install -r requirements.txt
```


##  Usage

### 1. Build Error Correction Datastore

```bash
bash codegen_sources/knnmt/create_datastore.sh
```

* This step preprocesses your training code and constructs a **key-value datastore** for KNN-based error correction.


### 2. Execute Code Error Correction

```bash
python execute.py 
```

**Key Arguments:**

* `--model` → Specify the base Transformer model (default: TransCoder-ST).
* `--datastore` → Path to the prebuilt error correction datastore.
* `--input` → Source code file to correct.

