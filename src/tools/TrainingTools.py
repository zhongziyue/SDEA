# import torchtext as tt
import sys

import torch as t
from torch.utils.data import DataLoader
from tqdm import tqdm

from tools.Announce import Announce


class TrainingTools:
    def __init__(self, iter: DataLoader, epochs: int = 15, device: t.device or str = 'cpu'):
        self.iter = iter
        self.sample_count = len(iter.dataset)
        self.epochs = epochs
        self.device = device

        self.__init_metrics()

    def __init_metrics(self):
        self.loss_count = 0
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0
        self.accuracy = 0
        self.precision = 0
        self.recall = 0
        self.f1_score = 0
        self.pred = t.LongTensor().to(self.device)
        self.results = t.FloatTensor().to(self.device)
        self.labels = t.LongTensor().to(self.device)

    def train(self):
        print(Announce.done(), '开始训练')
        print(Announce.printMessage(), 'Epochs:', self.epochs, 'Batch Size:', self.iter.batch_size, sep='\t')
        for epoch in range(1, self.epochs + 1):
            print(Announce.doing(), 'Epoch', epoch, '/', self.epochs, 'start')
            yield epoch
            print(Announce.done(), 'Epoch', epoch, '/', self.epochs, 'end')
        print(Announce.done(), '训练结束')

    def batches(self, get_bc):
        self.__init_metrics()
        with tqdm(total=self.sample_count, file=sys.stdout) as pbar:
            done = 0
            for i, batch in enumerate(self.iter):
                # batch: DataLoader
                yield i, batch
                # batch_size = len(batch[0][0])
                # batch_size = len(batch[0])
                batch_size = get_bc(batch)
                done += batch_size * 2
                pbar.set_description('Loss: %f\tAcc: %f\tPrec: %.2f\tRec: %.2f\tF1: %.2f' % (self.loss_count / done, self.accuracy, self.precision, self.recall, self.f1_score))
                # pbar.set_postfix(Prec=self.precision, Rec=self.recall, F1=self.f1_score)
                pbar.update(batch_size)

    @staticmethod
    def batch_iter(iter: DataLoader, desc=''):
        total = len(iter.dataset)
        with tqdm(total=total, file=sys.stdout) as pbar:
            done = 0
            pbar.set_description(desc)
            for i, batch in enumerate(iter):
                yield i, batch
                step_count = len(batch[0])
                done += step_count
                pbar.update(step_count)

    def update_results(self, y_pred: t.Tensor):
        # _, values = t.max(y_pred, 1)
        y_pred = y_pred.softmax(dim=1)
        values = y_pred[:, 1]
        # print(y_pred)
        # print(values)
        values = values.to(self.device)
        self.results = t.cat([self.results, values])

    def update_metrics(self, loss, y_pred, labels: t.Tensor=None, batch_size=None):
        assert batch_size is not None
        self.__add_loss_count(loss, batch_size)
        _, values = t.max(y_pred, 1)
        if labels is not None:
            tp, tn, fp, fn = TrainingTools._confusion_matrix(y_pred, labels)
            self.__update_classify_metrics(tp, tn, fp, fn)
        values = values.to(self.device)
        self.pred = t.cat([self.pred, values])
        if labels is not None:
            labels = labels.to(self.device)
            self.labels = t.cat([self.labels, labels])
        pass

    def __add_loss_count(self, loss, batch_size):
        self.loss_count += loss.item() * batch_size

    def __update_classify_metrics(self, TP, TN, FP, FN):
        self.TP += TP
        self.TN += TN
        self.FP += FP
        self.FN += FN
        self.precision = 100 * self.TP / (self.TP + self.FP)
        self.recall = 100 * self.TP / (self.TP + self.FN)
        self.f1_score = 2 * self.precision * self.recall / (self.precision + self.recall)
        self.accuracy = 100 * (self.TP + self.TN) / (self.TP + self.TN + self.FP + self.FN)

    def get_loss(self):
        return self.loss_count / self.sample_count

    def get_acc(self):
        return float(self.accuracy)

    def get_F1(self):
        return float(self.f1_score)

    def get_results(self):
        return list(self.results.cpu().numpy())

    @staticmethod
    def _confusion_matrix(output, target):
        predictions = output.max(1)[1].data
        correct = (predictions == target.data).float()
        incorrect = (1 - correct).float()
        positives = (target.data == 1).float()
        negatives = (target.data == 0).float()

        tp = t.dot(correct, positives)
        tn = t.dot(correct, negatives)
        fp = t.dot(incorrect, negatives)
        fn = t.dot(incorrect, positives)
        return tp, tn, fp, fn

    @staticmethod
    def pos_neg_count(y_true: t.Tensor, batch_size: int):
        pos_count = y_true.sum()
        pos_count = pos_count.item()
        neg_count = batch_size - pos_count
        return pos_count, neg_count

    @staticmethod
    def cal_loss(y_pred, y_true, pos_neg, device: t.device):
        loss = t.nn.CrossEntropyLoss(weight=t.FloatTensor(pos_neg).to(device), reduction='mean')(y_pred, y_true)
        return loss
