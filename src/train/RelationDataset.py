from torch.utils.data import Dataset
from preprocess.KBStore import KBStore
from train.RelationModel import RelationModel
import torch as t


class RelationDataset(Dataset):
    def __init__(self, train_tups_r, fs1: KBStore, fs2: KBStore, ent2embed1: t.Tensor, ent2embed2: t.Tensor):
        self.fs1 = fs1
        self.fs2 = fs2
        self.train_tups_r = train_tups_r
        self.ent2embed1 = ent2embed1.detach()
        self.ent2embed2 = ent2embed2.detach()
        self.all_neighbor1s, self.all_rel1s, self.nm_len1 = self.get_neighbor_matrix(fs1.facts)
        self.all_neighbor2s, self.all_rel2s, self.nm_len2 = self.get_neighbor_matrix(fs2.facts)
        self.pad_idx1 = self.ent2embed1.shape[0] - 1
        self.pad_idx2 = self.ent2embed2.shape[0] - 1
        self.pad_idxr1 = len(fs1.relation_ids)
        self.pad_idxr2 = len(fs2.relation_ids)
        print('pad_idxr1', self.pad_idxr1)
        print('pad_idxr2', self.pad_idxr2)
        print('nm_len1', self.nm_len1)
        print('nm_len2', self.nm_len2)
        print('pad_idx1', self.pad_idx1)
        print('pad_idx2', self.pad_idx2)

    def __len__(self):
        return len(self.train_tups_r)

    @staticmethod
    def get_neighbor_matrix(facts: dict):
        lens = [len(facts) if facts is not None else 0 for ent, facts in facts.items()]
        len_threshold = 1000
        N = min(max(lens), len_threshold)
        neighbors = {key: [ent for rel, ent in value] for key, value in facts.items()}
        rels = {key: [rel for rel, ent in value] for key, value in facts.items()}
        return neighbors, rels, N

    @staticmethod
    def get_matrix(neighbors, nm_len, pad_idx):
        if len(neighbors) > nm_len:
            neighbors = neighbors[:nm_len]
        matrix = t.tensor(neighbors + [pad_idx] * (nm_len - len(neighbors)), dtype=t.long)
        return matrix

    def __getitem__(self, item):
        pe1, pe2, ne1, ne2 = self.train_tups_r[item]
        pn1 = self.all_neighbor1s.get(pe1)
        pn2 = self.all_neighbor2s.get(pe2)
        nn1 = self.all_neighbor1s.get(ne1)
        nn2 = self.all_neighbor2s.get(ne2)

        pr1 = self.all_rel1s.get(pe1)
        pr2 = self.all_rel2s.get(pe2)
        nr1 = self.all_rel1s.get(ne1)
        nr2 = self.all_rel2s.get(ne2)

        return self.ent2embed1[pe1], self.ent2embed2[pe2], self.ent2embed1[ne1], self.ent2embed2[ne2], \
               self.get_matrix(pn1, self.nm_len1, self.pad_idx1), self.get_matrix(pn2, self.nm_len2, self.pad_idx2), self.get_matrix(nn1, self.nm_len1, self.pad_idx1), self.get_matrix(nn2, self.nm_len2, self.pad_idx2), \
               self.get_matrix(pr1, self.nm_len1, self.pad_idxr1), self.get_matrix(pr2, self.nm_len2, self.pad_idxr2), self.get_matrix(nr1, self.nm_len1, self.pad_idxr1), self.get_matrix(nr2, self.nm_len2, self.pad_idxr2)
