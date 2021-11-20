from collections import Counter
from itertools import chain

from transformers import BertTokenizer

from config.KBConfig import *
from tools import FileTools
from tools.Announce import Announce
from tools.MultiprocessingTool import MPTool


class BertDataLoader:
    def __init__(self, dataset):
        self.dataset = dataset

    def run(self):
        datas = self.load_data(BertDataLoader.line_to_feature)
        self.save_data(datas)
        self.save_token_freq(datas)

    def load_data(self, line_solver):
        data_path = self.dataset.seq_out
        print(Announce.doing(), 'load BertTokenizer')
        tokenizer = BertTokenizer.from_pretrained(args.pretrain_bert_path)
        print(Announce.done())
        with open(data_path, 'r', encoding='utf-8') as rfile:
            datas = MPTool.packed_solver(line_solver, tokenizer=tokenizer).send_packs(rfile).receive_results()
        return datas

    @staticmethod
    def line_to_feature(line: str, tokenizer: BertTokenizer):
        eid, text = line.strip('\n').split('\t')
        tokens = tokenizer.tokenize(text)
        tid_seq = tokenizer.convert_tokens_to_ids(tokens)
        return int(eid), tokens, tid_seq

    def save_data(self, datas):
        tokens_path = self.dataset.tokens_out
        tids_path = self.dataset.tids_out
        tids = [(eid, tids) for eid, tokens, tids in datas]
        tokens = [(eid, tokens) for eid, tokens, tids in datas]
        FileTools.save_list_p(tids, tids_path)
        FileTools.save_list_p(tokens, tokens_path)
        # return tokens, tids

    @staticmethod
    def load_saved_data(dataset):
        tids_path = dataset.tids_out
        tids = FileTools.load_list_p(tids_path)
        return tids

    def save_token_freq(self, datas) -> dict:
        freq_path = self.dataset.token_freqs_out
        tokens = [tokens for eid, tokens, tids in datas]
        tids = [tids for eid, tokens, tids in datas]
        tokens = list(chain.from_iterable(tokens))
        tids = list(chain.from_iterable(tids))
        results = [(token, tid) for token, tid in zip(tokens, tids)]
        r_counter = Counter(results)
        # FileTools.save_dict(r_counter, freq_path)
        r_dict = dict(r_counter)
        r_list = sorted(r_dict.items(), key=lambda x: x[1], reverse=True)
        FileTools.save_list_p(r_list, freq_path)
        return r_dict

    @staticmethod
    def load_freq(dataset):
        freq_path = dataset.token_freqs_out
        print(Announce.printMessage(), 'load:', freq_path)
        freq_list = FileTools.load_list_p(freq_path)
        freq_dict = {key: value for key, value in freq_list}
        return freq_dict
