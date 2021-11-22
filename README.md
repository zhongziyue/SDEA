# SDEA

## Dependencies
- python: 3.7
- pytorch: 1.3.1
- transformers: 4.2.2
- tqdm: 4.56.0

[comment]: <> (```)

[comment]: <> (pip install transformers)

[comment]: <> (```)


## Datasets Preparation
- SRPRS: https://github.com/nju-websoft/RSN/raw/master/entity-alignment-full-data.7z

- DBP15K: http://ws.nju.edu.cn/jape/data/DBP15k.tar.gz

1. Download the datasets and unzip them into _"SDEA/data"_. 

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
│   │   ├── en_de_15K_V1/
│   │   ├── en_fr_15K_V1/
│   │   ├── dbp_wd_15K_V1/
│   │   ├── dbp_yg_15K_V1/
```

2. Preprocess the datasets. 
```
cd src
python DBPDSPreprocess.py
python SRPRSPreprocess.py
```