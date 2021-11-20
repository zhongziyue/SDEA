import math
import sys

import torch as t
from tqdm import tqdm

from preprocess.KBStore import KBStore


class RelationValidDataset:
    def __init__(self, ents, fs: KBStore, all_embeds, batch_size):
        self.batch_size = batch_size
        self.fs = fs
        self.ents = ents
        self.all_embeds = all_embeds
        self.iter_count = 0
        self.batch_count = math.ceil(len(self.ents) / self.batch_size)
        self.tqdm = tqdm(total=self.batch_count, file=sys.stdout)

    def __iter__(self):
        return self

    def __next__(self):
        old_iter = self.iter_count
        self.iter_count += 1
        if self.iter_count < self.batch_count:
            results = self.ents[old_iter * self.batch_size: self.iter_count * self.batch_size]
        elif self.iter_count == self.batch_count:
            results = self.ents[old_iter * self.batch_size:]
        else:
            self.iter_count = 0
            self.tqdm.close()
            raise StopIteration()

        ents = []
        fs = []
        for ent in results:
            ents.append(self.all_embeds[ent])
            facts = self.fs.facts.get(ent)
            fs.append(facts)
        ents = t.stack(ents, dim=0)
        self.tqdm.update()
        return ents, fs
