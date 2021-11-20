from preprocess.BertDataLoader import BertDataLoader
from config.KBConfig import *
from preprocess.KBStore import KBStore
from tools.Announce import Announce
from train.PairwiseTrainer import PairwiseTrainer
import torch as t
from train.RelationTrainer import RelationTrainer

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    # device = t.device("cuda" if t.cuda.is_available() else "cpu")
    device = t.device('cuda')
    print(device)
    tids1 = BertDataLoader.load_saved_data(dataset1)
    tids2 = BertDataLoader.load_saved_data(dataset2)

    max_len1 = max([len(tokens) for eid, tokens in tids1])
    max_len2 = max([len(tokens) for eid, tokens in tids2])
    print(Announce.printMessage(), 'Max len 1:', max_len1)
    print(Announce.printMessage(), 'Max len 2:', max_len2)
    eid2tids1 = {eid: tids for eid, tids in tids1}
    eid2tids2 = {eid: tids for eid, tids in tids2}
    fs1 = KBStore(dataset1)
    fs2 = KBStore(dataset2)
    # fs1.load_entities()
    # fs2.load_entities()
    fs1.load_kb_from_saved()
    fs2.load_kb_from_saved()
    if args.basic_bert_path is None:
        trainer = PairwiseTrainer()
        trainer.data_prepare(eid2tids1, eid2tids2, fs1, fs2)
        trainer.train(device=device)

    rel_trainer = RelationTrainer()
    rel_trainer.data_prepare(eid2tids1, eid2tids2, fs1, fs2)
    if args.basic_bert_path is None:
        bert_model_path = links.model_save
    else:
        bert_model_path = args.basic_bert_path
    rel_trainer.train(bert_model_path, device=device)
