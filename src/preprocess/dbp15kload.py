import random

from config.DBPConfig import *
from preprocess import Parser
from tools import FileTools
from tools.Announce import Announce
from tools.MultiprocessingTool import MultiprocessingTool


class dbp15kload:
    def __init__(self):
        pass

    def run(self):
        if not os.path.exists(dataset1.attr):
            os.makedirs(dataset1.attr)
        if not os.path.exists(dataset2.attr):
            os.makedirs(dataset2.attr)

        ent_name_1 = self.read_entities(dbp1.eid_path)
        ent_name_2 = self.read_entities(dbp2.eid_path)
        rel_name_1 = self.read_entities(dbp1.rid_path)
        rel_name_2 = self.read_entities(dbp2.rid_path)
        train_links = self.read_links(dbplinks.train_links_path, ent_name_1, ent_name_2)
        test_links = self.read_links(dbplinks.test_links_path, ent_name_1, ent_name_2)
        ent_links_name = [*train_links, *test_links]
        random.shuffle(train_links)
        train_count = 2 * len(train_links) // 3
        # train_count = len(train_links)
        train_train_links = train_links[:train_count]
        train_valid_links = train_links[train_count:]
        fold_path = os.path.dirname(links.train)
        if not os.path.exists(fold_path):
            os.makedirs(fold_path)
        FileTools.save_list(train_train_links, links.train)
        FileTools.save_list(train_valid_links, links.valid)
        FileTools.save_list(test_links, links.test)
        FileTools.save_list(ent_links_name, links.truth)
        
        rel_triples_1 = self.read_id_triples(dbp1.rel_path)
        rel_triples_2 = self.read_id_triples(dbp2.rel_path)
        rel_triples_name_1 = [(ent_name_1.get(sid), rel_name_1.get(rid), ent_name_1.get(oid)) for sid, rid, oid in rel_triples_1]
        rel_triples_name_2 = [(ent_name_2.get(sid), rel_name_2.get(rid), ent_name_2.get(oid)) for sid, rid, oid in rel_triples_2]
        # rel_triples_name_1 = filter(lambda tup: tup[0] in entities_1 and tup[2] in entities_1, rel_triples_name_1)
        # rel_triples_name_2 = filter(lambda tup: tup[0] in entities_2 and tup[2] in entities_2, rel_triples_name_2)
        FileTools.save_list(rel_triples_name_1, dataset1.rel)
        FileTools.save_list(rel_triples_name_2, dataset2.rel)

        att_triples_1 = Parser.for_file(dbp1.attr_path, Parser.OEAFileType.ttl_full)
        att_triples_2 = Parser.for_file(dbp2.attr_path, Parser.OEAFileType.ttl_full)
        # att_triples_1 = filter(lambda tup: tup[0] in entities_1, att_triples_1)
        # att_triples_2 = filter(lambda tup: tup[0] in entities_2, att_triples_2)
        FileTools.save_list(att_triples_1, '/'.join((dataset1.attr, '2attr')))
        FileTools.save_list(att_triples_2, '/'.join((dataset2.attr, '2attr')))

        entities_1 = {e1 for e1, e2 in ent_links_name} | {e1 for e1, r, e2 in rel_triples_name_1} | {e2 for e1, r, e2 in rel_triples_name_1} | {e for e, a, l in att_triples_1}
        entities_2 = {e2 for e1, e2 in ent_links_name} | {e1 for e1, r, e2 in rel_triples_name_2} | {e2 for e1, r, e2 in rel_triples_name_2} | {e for e, a, l in att_triples_2}
        print('len entities_1', len(entities_1))
        print('len entities_2', len(entities_2))

        # name_attr_1 = [(e, 'BertIntName', e[e.index('dbpedia.org/resource/')+len('dbpedia.org/resource/'):])for e in entities_1]
        # name_attr_2 = [(e, 'BertIntName', e[e.index('dbpedia.org/resource/')+len('dbpedia.org/resource/'):])for e in entities_2]
        # FileTools.save_list(name_attr_1, '/'.join((dataset1.attr, '1name')))
        # FileTools.save_list(name_attr_2, '/'.join((dataset2.attr, '1name')))

        import pickle
        desc_dict: dict = pickle.load(open(desc_path, "rb"))
        # desc_list_1 = self.get_desc(desc_dict.items(), entities_1, dbp1.name)
        # desc_list_2 = self.get_desc(desc_dict.items(), entities_2, dbp2.name)
        # FileTools.save_list(desc_list_1, '/'.join((dataset1.attr, '3desc')))
        # FileTools.save_list(desc_list_2, '/'.join((dataset2.attr, '3desc')))

    @staticmethod
    def read_entities(file):
        print(Announce.printMessage(), 'read', file)
        with open(file, 'r', encoding='utf-8') as rfile:
            mt = MultiprocessingTool(num_workers=20)
            ent_list = mt.packed_solver(lambda line: line.strip('\n').split('\t')).send_packs(rfile).receive_results()
            entity_names = {int(eid): ent for eid, ent in ent_list}
        return entity_names

    @staticmethod
    def read_id_triples(file):
        print(Announce.printMessage(), 'read', file)
        with open(file, 'r', encoding='utf-8') as rfile:
            mt = MultiprocessingTool(num_workers=20)
            list = mt.packed_solver(lambda line: line.strip('\n').split('\t')).send_packs(rfile).receive_results()
            triples = [(int(tid) for tid in triple) for triple in list]
        return triples

    @staticmethod
    def read_links(file, ent_names_1, ent_names_2):
        def links_line(line: str):
            sbj, obj = line.strip('\n').split('\t')
            sbj = int(sbj)
            obj = int(obj)
            sbj, obj = ent_names_1[sbj], ent_names_2[obj]
            return sbj, obj
        with open(file, 'r', encoding='utf-8') as rfile:
            mt = MultiprocessingTool(num_workers=20)
            link_list = mt.packed_solver(links_line).send_packs(rfile).receive_results()
        return link_list

    @staticmethod
    def get_desc(desc_list, ent_names: set, name):
        def desc_line(line):
            ent, desc = line
            if ent.startswith(dbp15kload.get_prefix(name)) and ent in ent_names:
            # if ent.startswith(dbp15kload.get_prefix(name)):
                return (ent, 'BertIntDesc', desc)
            return (None, None, None)

        mt = MultiprocessingTool(num_workers=20)
        results = mt.packed_solver(desc_line).send_packs(desc_list).receive_results()
        results = [tup for tup in results if tup[0] is not None]
        return results
        # return [(ent, 'BertIntDesc', desc) for ent, desc in desc_list if ent.startswith(prefix)]

    @staticmethod
    def get_prefix(name):
        if name == 'zh':
            return 'http://zh.dbpedia.org/'
        if name == 'en':
            return 'http://dbpedia.org/'
        if name == 'fr':
            return 'http://fr.dbpedia.org/'
        if name == 'ja':
            return 'http://ja.dbpedia.org/'
    pass
