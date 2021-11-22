cd src

python SRPRSPreprocess.py

version="SDEA"
gpus='0'

# dbp_wd_15k_V1
paras=""
paras="$paras --datasets_root ../data"
paras="$paras --dataset dbp_wd_15k_V1"
paras="$paras --result_root ../outputs"
paras="$paras --pretrain_bert_path ../pre_trained_models/bert-base-multilingual-uncased"
paras="$paras --log"
paras="$paras --relation"
paras="$paras --version ${version}"
echo $paras
python -u SDEAPreprocess.py $paras
paras="$paras --fold 0"
paras="$paras --gpus ${gpus}"
python -u SDEATrain.py $paras


# dbp_yg_15k_V1
paras=""
paras="$paras --datasets_root ../data"
paras="$paras --dataset dbp_yg_15k_V1"
paras="$paras --result_root ../outputs"
paras="$paras --pretrain_bert_path ../pre_trained_models/bert-base-multilingual-uncased"
paras="$paras --log"
paras="$paras --relation"
paras="$paras --version ${version}"
echo $paras
python -u SDEAPreprocess.py $paras
paras="$paras --fold 0"
paras="$paras --gpus ${gpus}"
python -u SDEATrain.py $paras


# en_de_15k_V1
paras=""
paras="$paras --datasets_root ../data"
paras="$paras --dataset en_de_15k_V1"
paras="$paras --result_root ../outputs"
paras="$paras --pretrain_bert_path ../pre_trained_models/bert-base-multilingual-uncased"
paras="$paras --log"
paras="$paras --relation"
paras="$paras --version ${version}"
echo $paras
python -u SDEAPreprocess.py $paras
paras="$paras --fold 0"
paras="$paras --gpus ${gpus}"
python -u SDEATrain.py $paras


# en_fr_15k_V1
paras=""
paras="$paras --datasets_root ../data"
paras="$paras --dataset en_fr_15k_V1"
paras="$paras --result_root ../outputs"
paras="$paras --pretrain_bert_path ../pre_trained_models/bert-base-multilingual-uncased"
paras="$paras --log"
paras="$paras --relation"
paras="$paras --version ${version}"
echo $paras
python -u SDEAPreprocess.py $paras
paras="$paras --fold 0"
paras="$paras --gpus ${gpus}"
python -u SDEATrain.py $paras
