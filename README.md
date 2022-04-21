# SDEA

## Dependencies
- python: 3.7
- pytorch: 1.3.1
- transformers: 4.2.2
- tqdm: 4.56.0

## Installation
We recommend creating a new conda environment to install the dependencies and run SDEA:

```shell
conda create -n SDEA python=3.7
conda activate SDEA
conda install pytorch-gpu=1.3.1
pip install transformers==4.2.2
```

## Preparation

The structure of the project is listed as follows:

```
SDEA/
├── src/: The soruce code of SDEA. 
├── data/: The datasets. 
│   ├── DBP15k/: The downloaded DBP15K benchmark. 
│   │   ├── fr_en/
│   │   ├── ja_en/
│   │   ├── zh_en/
│   ├── entity-alignment-full-data/: The downloaded SRPRS benchmark. 
│   │   ├── en_de_15k_V1/
│   │   ├── en_fr_15k_V1/
│   │   ├── dbp_wd_15k_V1/
│   │   ├── dbp_yg_15k_V1/
├── pre_trained_models/: The pre-trained transformer-based models. 
│   ├── bert-base-multilingual-uncased: The model used in our experiments.
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   ├── tokenizer.json
│   │   ├── tokenizer_config.json
│   │   ├── vocab.txt
│   ├── ......
```

### Datasets

- SRPRS: https://github.com/nju-websoft/RSN/raw/master/entity-alignment-full-data.7z

- DBP15K: ~~http://ws.nju.edu.cn/jape/data/DBP15k.tar.gz~~ Service unavailable now. Please download from [Google Drive](https://drive.google.com/file/d/1Xj6CaeECTDwuJM5nj_Xk3JZt_oXFu5sO/view?usp=sharing).

1. Download the datasets and unzip them into _"SDEA/data"_.

2. Preprocess the datasets.

```
cd src
python DBPDSPreprocess.py
python SRPRSPreprocess.py
```

### Pre-trained Models

The pre-trained models of _transformers_ library can be downloaded from https://huggingface.co/models. 
We use [bert-base-multilingual-uncased](https://huggingface.co/bert-base-multilingual-uncased) in our experiments. 

Please put the downloaded pre-trained models into _"SDEA/pre_trained_models"_. 


## How to Run

```shell
bash run_dbp15k.sh
bash run_SRPRS.sh
```