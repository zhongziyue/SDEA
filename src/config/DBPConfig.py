import argparse
import os
import random
import torch as t
import numpy as np
from config.KBConfig import Dataset, OEAlinks, args

seed = 11037
random.seed(seed)
t.manual_seed(seed)
t.cuda.manual_seed_all(seed)
np.random.seed(seed)

# parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', type=str)
# args = parser.parse_args()

dbp15k_root = '/home/zhongziyue/db_integration/code/pc_bert_int/data/dbp15k'
dbp15k_path = '/'.join((dbp15k_root, args.dataset_name))
dataset_root = '/home/zhongziyue/db_integration/datasets/dbp15k_for_GEA'
desc_path = '/home/zhongziyue/db_integration/code/pc_bert_int/data/dbp15k/2016-10-des_dict'
names = args.dataset_name.split('_')


# class DBPDataset:
#     def __init__(self):
#         triples_1 = names[0] + '_att_triples'
#         triples_2 = names[1] + '_att_triples'
#
#         self.triples_path_1 = os.path.join(dbp15k_path, triples_1)
#         self.triples_path_2 = os.path.join(dbp15k_path, triples_2)
#         self.rel_path_1 = os.path.join(dbp15k_path, 'triples_1')
#         self.rel_path_2 = os.path.join(dbp15k_path, 'triples_2')
#         self.rid_path_1 = os.path.join(dbp15k_path, 'rel_ids_1')
#         self.rid_path_2 = os.path.join(dbp15k_path, 'rel_ids_2')
#         self.eid_path_1 = os.path.join(dbp15k_path, 'ent_ids_1')
#         self.eid_path_2 = os.path.join(dbp15k_path, 'ent_ids_2')
#         self.train_links_path = os.path.join(dbp15k_path, 'sup_pairs')
#         self.test_links_path = os.path.join(dbp15k_path, 'ref_pairs')
#         pass

class DBPDataset:
    def __init__(self, no):
        self.name = names[no-1]
        triples = self.name + '_att_triples'

        self.attr_path = os.path.join(dbp15k_path, triples)
        self.rel_path = self.paths('triples', no)
        self.rid_path = self.paths('rel_ids', no)
        self.eid_path = self.paths('ent_ids', no)
        pass

    @staticmethod
    def paths(name, no=None):
        if no is None:
            file_name = name
        else:
            file_name = '_'.join((name, str(no)))
        return os.path.join(dbp15k_path, file_name)


class DBPlinks:
    def __init__(self):
        self.train_links_path = DBPDataset.paths('sup_pairs')
        self.test_links_path = DBPDataset.paths('ref_pairs')


dbp1 = DBPDataset(1)
dbp2 = DBPDataset(2)
dataset1 = Dataset(1)
dataset2 = Dataset(2)
dbplinks = DBPlinks()
links = OEAlinks(args.fold)
