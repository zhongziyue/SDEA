from tools.Announce import Announce
from config.KBConfig import *


class Trainer:
    def __init__(self):
        pass

    def train(self):
        pass

    def data_prepare(
            self, eid2tids1: dict, eid2tids2: dict,
            entity_ids1: dict, entity_ids2: dict,
    ):
        pass

    @staticmethod
    def data_item_solver(
            pair,
            eid2tids1, eid2tids2,
            cls_token, sep_token,
            pad_tolen=0,
            **kwargs
    ):
        pass

    @staticmethod
    def reduce_tokens(tids):
        while True:
            total_length = len(tids)
            if total_length <= seq_max_len:
                break
            tids.pop()
        return tids


