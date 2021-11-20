from torch.utils.data import Dataset


class PairwiseDataset(Dataset):
    def __init__(self, train_tups, ent2data1, ent2data2):
        self.train_tups = train_tups
        self.ent2data1 = ent2data1
        self.ent2data2 = ent2data2

    def __len__(self):
        return len(self.train_tups)

    def __getitem__(self, item):
        pe1, pe2, ne1, ne2 = self.train_tups[item]
        return self.ent2data1[pe1], self.ent2data2[pe2], self.ent2data1[ne1], self.ent2data2[ne2]
