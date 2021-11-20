import csv
import re
from collections import defaultdict
from typing import List, Iterator

from tqdm import tqdm

from config.KBConfig import *
from preprocess import Parser
from preprocess.Parser import OEAFileType
from tools import FileTools
from tools.Announce import Announce
from tools.MultiprocessingTool import MPTool
from tools.MyTimer import MyTimer
from tools.text_to_word_sequence import text_to_word_sequence


class KBStore:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.entities = []
        # self.classes = self.entities
        # self.literals = self.entities
        self.literals = []
        self.entity_ids = {}
        self.classes_ids = {}
        self.literal_ids = {}

        self.relations = []
        # self.properties = self.relations
        self.properties = []
        self.relation_ids = {}
        self.property_ids = {}

        self.words = []
        self.word_ids = {}

        self.facts = {}
        self.literal_facts = {}
        self.blocks = {}
        self.word_level_blocks = {}

        self.properties_functionality = None
        self.relations_functionality = None

    def load_kb(self) -> None:
        timer = MyTimer()
        # load attr
        self.load_path(self.dataset.attr, self.load, OEAFileType.attr)
        if args.relation:
            self.load_path(self.dataset.rel, self.load, OEAFileType.rel)
            self.relations_functionality = KBStore.calculate_func(self.relations, self.relation_ids, self.facts, self.entity_ids)
        # 增加对facts按relation排序
        for ent, facts in self.facts.items():
            facts: list
            facts.sort(key=lambda x: (x[0], x[1]), reverse=False)
        self.properties_functionality = KBStore.calculate_func(self.properties, self.property_ids, self.literal_facts,
                                                               self.entity_ids)
        timer.stop()

        print(Announce.printMessage(), 'Finished loading in', timer.total_time())
        self.save_base_info()
        self.save_datas()

    def save_base_info(self):
        print(Announce.doing(), 'Save Base Info')
        FileTools.save_dict_reverse(self.entity_ids, self.dataset.entities_out)

        FileTools.save_dict_reverse(self.literal_ids, self.dataset.literals_out)
        print(Announce.done(), 'Finished Saving Base Info')

    def save_property_table(self):
        table_path = self.dataset.table_out
        print(Announce.doing(), 'Save', table_path)
        with open(table_path, 'w', encoding='utf-8') as fp:
            header = ['id', 'ent_name']
            if not functionality_control:
                header.extend(self.property_ids.keys())
            else:
                for p, pid in self.property_ids.items():
                    if self.properties_functionality[pid] > functionality_threshold:
                        header.append(p)
            writer = csv.DictWriter(fp, header)
            writer.writeheader()
            dicts = MPTool.packed_solver(self.get_property_table_line).send_packs(self.entity_ids.items()).receive_results()
            # dicts = [dic for dic in dicts if dicts is not None]
            dicts = filter(lambda dic: dic is not None, dicts)
            dicts = list(dicts)
            for dic in dicts:
                writer.writerow(dic)
        print(Announce.done())
        return dicts, header

    def save_seq_form(self, dicts: Iterator, header: List):
        def get_seq(dic: dict):
            eid = dic['id']
            values = [dic[key] for key in header if key in dic]
            seq = ' '.join(values)
            assert len(seq) > 0
            return eid, seq
        seq_path = self.dataset.seq_out
        print(Announce.doing(), 'Save', seq_path)
        header = header.copy()
        header.remove('id')
        seqs = MPTool.packed_solver(get_seq).send_packs(dicts).receive_results()
        # seqs = [get_seq(dic) for dic in dicts]
        FileTools.save_list(seqs, seq_path)
        print(Announce.done())

    def save_facts(self):
        print(Announce.doing(), 'Save facts')
        FileTools.save_dict_p(self.facts, self.dataset.facts_out)
        print(Announce.done(), 'Save facts')

    def save_datas(self):
        print(Announce.doing(), 'Save data2')
        print(Announce.printMessage(), 'Save', self.dataset.properties_out)
        with open(self.dataset.properties_out, 'w', encoding='utf-8') as writer:
            for r, id in self.property_ids.items():
                print(id, r, self.properties_functionality[id], sep='\t', file=writer)

        if args.relation:
            print(Announce.printMessage(), 'Save', self.dataset.relations_out)
            with open(self.dataset.relations_out, 'w', encoding='utf-8') as wfile:
                for r, id in self.relation_ids.items():
                    print((id, r, self.relations_functionality[id]), file=wfile)

        # 保存property csv
        dicts, header = self.save_property_table()
        self.save_seq_form(dicts, header)
        if args.relation:
            self.save_facts()
        print(Announce.done(), 'Finished')

    def load_kb_from_saved(self):
        self.load_entities()
        self.load_literals()
        self.load_relations()
        self.load_properties()
        self.load_facts()
        pass

    def load_entities(self):
        print(Announce.doing(), 'Load entities', self.dataset.entities_out)
        # self.entity_ids = FileTools.load_dict_reverse(self.dataset.entities_out)
        entity_list = FileTools.load_list(self.dataset.entities_out)
        self.entity_ids = {ent: int(s_eid) for s_eid, ent in entity_list}
        self.entities = [ent for s_eid, ent in entity_list]
        print(Announce.done())
        pass

    def load_relations(self):
        relation_list = FileTools.load_list_p(self.dataset.relations_out)
        self.relation_ids = {rel: rid for rid, rel, func in relation_list}
        self.relations = [rel for rid, rel, func in relation_list]

    def load_literals(self):
        entity_list = FileTools.load_list(self.dataset.literals_out)
        self.literal_ids = {ent: int(s_eid) for s_eid, ent in entity_list}

    def load_properties(self):
        property_list = FileTools.load_list(self.dataset.properties_out)
        self.property_ids = {prop: s_pid for s_pid, prop, s_func in property_list}

    def load_facts(self):
        fact_list = FileTools.load_list_p(self.dataset.facts_out)
        self.facts = {eid: elist for eid, elist in fact_list}

    @staticmethod
    def save_blocks(fs1, fs2):
        if args.blocking:

            pass
        pass

    # @staticmethod
    # def not_blocking(fs1, fs2):
    #     print(Announce.printMessage(), 'not blocking')
    #     fs1: KBStore
    #     fs2: KBStore
    #     print(Announce.printMessage(), 'get all entities')
    #     e1s = set(fs1.entity_ids.values())
    #     e2s = set(fs2.entity_ids.values())
    #     print(Announce.doing(), 'save all entities as a whole block')
    #     with open(links.block, 'w', encoding='utf-8') as wfile:
    #         print((e1s, e2s), file=wfile)
    #     print(Announce.done())

    def get_property_table_line(self, line):
        e, ei = line
        e: str
        ename = e.split('/')[-1]
        dic = {'id': ei, 'ent_name': ename}
        facts = self.literal_facts.get(ei)
        # if facts is None:
        #     return None
        if facts is not None:
            fact_aggregation = defaultdict(list)
            for fact in facts:
                # 过滤函数性低的
                if functionality_control and self.properties_functionality[fact[0]] <= functionality_threshold:
                    continue
                fact_aggregation[fact[0]].append(self.literals[fact[1]])

            for pid, objs in fact_aggregation.items():
                pred = self.properties[pid]
                obj = ' '.join(objs)
                dic[pred] = obj

        return dic
        # writer.writerow(dic)
        pass

    @staticmethod
    def load_path(path, load_func, file_type: OEAFileType) -> None:
        print(Announce.doing(), 'Start loading', path)
        if os.path.isdir(path):
            for file in sorted(os.listdir(path)):
                if os.path.isdir(file):
                    continue
                file = os.path.join(path, file)
                # load_func(file, type)
                KBStore.load_path(file, load_func, file_type)
        else:
            load_func(path, file_type)
        print(Announce.done(), 'Finish loading', path)

    def load(self, file: str, file_type: OEAFileType) -> None:
        tuples = Parser.for_file(file, file_type)
        with tqdm(desc='add tuples', file=sys.stdout) as tqdm_add:
            tqdm_add.total = len(tuples)
            for args in tuples:
                self.add_tuple(*args, file_type)
                tqdm_add.update()
        pass

    def add_tuple(self, sbj: str, pred: str, obj: str, file_type: OEAFileType) -> None:
        assert sbj is not None and obj is not None and pred is not None, 'sbj, obj, pred None'
        if file_type == OEAFileType.attr:
            if obj.startswith('"'):
                obj = obj[1:-1]
            toks = text_to_word_sequence(obj)
            for tok in toks:
                if len(tok) < 5:
                    continue
                if bool(re.search(r'\d', tok)):
                    return
            sbj_id = self.get_or_add_item(sbj, self.entities, self.entity_ids)
            obj_id = self.get_or_add_item(obj, self.literals, self.literal_ids)
            pred_id = self.get_or_add_item(pred, self.properties, self.property_ids)
            self.add_fact(sbj_id, pred_id, obj_id, self.literal_facts)
            self.add_to_blocks(sbj_id, obj_id)
            words = text_to_word_sequence(obj)
            self.add_word_level_blocks(sbj_id, words)
        elif file_type == OEAFileType.rel:
            sbj_id = self.get_or_add_item(sbj, self.entities, self.entity_ids)
            obj_id = self.get_or_add_item(obj, self.entities, self.entity_ids)
            pred_id = self.get_or_add_item(pred, self.relations, self.relation_ids)
            pred2_id = self.get_or_add_item(pred + '-', self.relations, self.relation_ids)
            self.add_fact(sbj_id, pred_id, obj_id, self.facts)
            self.add_fact(obj_id, pred2_id, sbj_id, self.facts)

    def add_item(self, name: str, names: list, ids: dict) -> int:
        iid = len(names)
        names.append(name)
        ids[name] = iid
        return iid

    def get_or_add_item(self, name: str, names: list, ids: dict) -> int:
        if name in ids:
            return ids.get(name)
        else:
            return self.add_item(name, names, ids)

    def add_fact(self, sbj_id, pred_id, obj_id, facts_list: dict) -> None:
        if sbj_id in facts_list:
            facts: list = facts_list.get(sbj_id)
            facts.append((pred_id, obj_id))
        else:
            facts_list[sbj_id] = [(pred_id, obj_id)]

    def add_to_blocks(self, sbj_id, obj_id) -> None:
        if obj_id in self.blocks:
            block: set = self.blocks.get(obj_id)
            block.add(sbj_id)
        else:
            self.blocks[obj_id] = {sbj_id}

    def add_word_level_blocks(self, entity_id, words):
        for word in words:
            if word in self.word_level_blocks:
                block: set = self.word_level_blocks.get(word)
                block.add(entity_id)
            else:
                self.word_level_blocks[word] = {entity_id}
        pass

    @staticmethod
    def calculate_func(r_names: list, r_ids: dict, facts_list: dict, sbj_ids: dict) -> list:
        num_occurrences = [0] * len(r_names)
        func = [0.] * len(r_names)
        num_subjects_per_relation = [0] * len(r_names)
        last_subject = [-1] * len(r_names)

        for sbj_id in sbj_ids.values():
            facts = facts_list.get(sbj_id)
            if facts is None:
                continue
            for fact in facts:
                num_occurrences[fact[0]] += 1
                if last_subject[fact[0]] != sbj_id:
                    last_subject[fact[0]] = sbj_id
                    num_subjects_per_relation[fact[0]] += 1

        for r_name, rid in r_ids.items():
            func[rid] = num_subjects_per_relation[rid] / num_occurrences[rid]
            print(Announce.printMessage(), rid, r_name, func[rid], sep='\t')
        return func
