import random
import sys

from preprocess import Parser
from config.KBConfig import *
from tools.Announce import Announce


class OEATrainPairGenerator:
    def __init__(self, name1, name2):
        self.name1 = name1
        self.name2 = name2

    def data_preprocessing(self, entity_ids_1, entity_ids_2, fold_num):
        verbose = self._verbose
        self._verbose = False
        truth_path = links.truth

        print(Announce.printMessage(), 'Truth')
        truth_list = self.read_truths(truth_path, entity_ids_1, entity_ids_2)
        truth_set = set(truth_list)
        print('len truth_list:', len(truth_list))
        print('len truth_set:', len(truth_set))
        # train_block_path = Alignment.get_block_path(args.train_block)
        # valid_block_path = Alignment.get_block_path(args.valid_block)
        # test_block_path = Alignment.get_block_path(args.valid_block)

        print(Announce.printMessage(), 'train_block_path:', train_block_path)
        print(Announce.printMessage(), 'valid_block_path:', valid_block_path)
        print(Announce.printMessage(), 'test_block_gen_path:', test_block_path)
        train_pair_candidates = self.load_blocks(train_block_path)
        valid_pair_candidates = self.load_blocks(valid_block_path)
        test_pair_candidates = self.load_blocks(test_block_path)
        # train_pair_candidates = self.load_candidates(train_block_path)
        # valid_pair_candidates = self.load_candidates(valid_block_path)

        print(Announce.printMessage(), 'Train')
        self.load_train(entity_ids_1, entity_ids_2, truth_set, 'train', fold_num, train_pair_candidates)

        print(Announce.printMessage(), 'Valid')
        self.load_train(entity_ids_1, entity_ids_2, truth_set, 'valid', fold_num, valid_pair_candidates)

        print(Announce.printMessage(), 'Test Block Gen')
        test_path = KBConfig.init_trains_path('test', fold_num)
        test_truths = self.read_truths(test_path, entity_ids_1, entity_ids_2)
        e1s = {e1 for e1, e2 in test_truths}
        e2s = {e2 for e1, e2 in test_truths}
        test_candidates = []
        for sbj, objs in test_pair_candidates.items():
            if sbj not in e1s:
                continue
            objs = set(objs)
            objs &= e2s
            test_candidates.append(({sbj}, objs))
        FileTools.save_list(test_candidates, KBConfig.test_block_path(fold_num))

        self._verbose = verbose

    def load_train(
            self, entity_ids_1, entity_ids_2,
            truth_set, train_type, fold_num,
            pair_candidates,
    ):
        # truth_id_pairs = {entities1.get(sbj): entities2.get(obj) for sbj, pred, obj in truths if sbj in entities1 and obj in entities2}
        # truth_id_pairs = {(entity_ids_1.get(sbj), entity_ids_2.get(obj)) for sbj, obj in truth_set if sbj in entity_ids_1 and obj in entity_ids_2}
        train_path = KBConfig.init_trains_path(train_type, fold_num)
        trains = self.read_truths(train_path, entity_ids_1, entity_ids_2)
        train_out_path = FileNames.txt_save_path('_'.join((self.name1, self.name2, str(fold_num))), '_'.join((train_type, 'data')))
        # train_set = set(trains)
        self.save_train_pairs(trains, truth_set, train_out_path, pair_candidates)
        # return truth_id_pairs

    def read_truths(self, link_path, entity_ids_1, entity_ids_2):
        with open(link_path, 'r', encoding='utf-8') as rfile:
            link_list = self.packed_solver(
                self.truth_solver,
                entity_ids_1=entity_ids_1,
                entity_ids_2=entity_ids_2,
            ).send_packs(rfile).receive_results()
        none_count = sum([1 for sbj, obj in link_list if sbj is None or obj is None])
        total_count = len(link_list)
        print('total_count:', total_count)
        print('none_count:', none_count)
        link_list = [(sbj, obj) for sbj, obj in link_list if sbj is not None and obj is not None]
        print('not none count:', len(link_list))
        return link_list

    def read_trains(self, link_path, entity_ids_1, entity_ids_2, truth_set):
        with open(link_path, 'r', encoding='utf-8') as rfile:
            link_list = self.packed_solver(
                self.train_solver,
                entity_ids_1=entity_ids_1,
                entity_ids_2=entity_ids_2,
                truth_set=truth_set,
            ).send_packs(rfile).receive_results()
        none_count = sum([1 for sbj, obj, label in link_list if sbj is None or obj is None])
        total_count = len(link_list)
        print('total_count:', total_count)
        print('none_count:', none_count)
        print('not none count:', total_count - none_count)
        return link_list

    @staticmethod
    def truth_solver(line: str, entity_ids_1: dict, entity_ids_2: dict):
        sbj, obj = line.strip().split('\t')
        sbj = Parser.compressUri(sbj)
        obj = Parser.compressUri(obj)
        sbj = entity_ids_1.get(sbj)
        obj = entity_ids_2.get(obj)
        return sbj, obj

    @staticmethod
    def train_solver(line: str, entity_ids_1: dict, entity_ids_2: dict, truth_set: set):
        sbj, obj = line.strip().split('\t')
        sbj = Parser.compressUri(sbj)
        obj = Parser.compressUri(obj)
        sbj = entity_ids_1.get(sbj)
        obj = entity_ids_2.get(obj)
        label = 1 if (sbj, obj) in truth_set else 0
        return sbj, obj, label

    @staticmethod
    def load_blocks(block_path):
        # block_path = FileNames.txt_save_path('_'.join((name1, name2)), 'blocks_reduce')
        pair_candidate = {}
        with open(block_path, 'r', encoding='utf-8') as fp:
            for line in fp:
                sbjs, objs = eval(line)
                for sbj in sbjs:
                    for obj in objs:
                        if sbj in pair_candidate:
                            obj_list = pair_candidate.get(sbj)
                            obj_list.append(obj)
                        else:
                            obj_list = [obj]
                            pair_candidate[sbj] = obj_list
        return pair_candidate

    def load_candidates(self, block_path):
        print(Announce.printMessage(), 'load candidates from', block_path)
        with open(block_path, 'r', encoding='utf-8') as rfile:
            candidates_list = self.packed_solver(self.blocks_line).send_packs(rfile).receive_results()
            candidates = {}
            print(Announce.printMessage(), '整理candidates')
            for cands, count in candidates_list:
                for sbj, obj_set in cands.items():
                    if sbj in candidates:
                        candidates[sbj].append(obj_set)
                    else:
                        candidates[sbj] = [obj_set]
            print(Announce.printMessage(), '集成candidates')
            candidates = {sbj: {obj for obj_set in obj_set_list for obj in obj_set} for sbj, obj_set_list in candidates.items()}
        return candidates

    @staticmethod
    def blocks_line(line):
        candidates = {}
        sbjs, objs = eval(line)
        for sbj in sbjs:
            if sbj in candidates:
                candidates[sbj] |= objs
            else:
                candidates[sbj] = set(objs)
        return candidates

    @staticmethod
    def save_train_pairs(trains, truth_set: set, train_out_path: str, pair_candidates):
        print(Announce.doing(), '构建sbj->objs映射列表')
        pairs = []
        print(Announce.printMessage(), '真值数量: ', len(truth_set))
        print(Announce.printMessage(), 'trains数量: ', len(trains))
        print(len(trains[0]))
        print(trains[0])
        for sbj, obj in trains:
            pairs.append((sbj, obj, 1))
            # 从blocks中挑一个
            obj_list = pair_candidates.get(sbj)
            if obj_list is None:
                continue
            j = random.randint(0, len(obj_list) - 1)
            obj2 = obj_list[j]
            # if obj2 != obj:
            if (sbj, obj2) not in truth_set:
                pairs.append((sbj, obj2, 0))
                pass
            else:
                for obj2 in obj_list:
                    if (sbj, obj2) not in truth_set:
                        pairs.append((sbj, obj2, 0))
                        break
            pass
        print(Announce.done(), '完成')
        print(Announce.doing(), '保存pair')
        FileTools.save_list(pairs, train_out_path)
        print(Announce.done(), '完成')
