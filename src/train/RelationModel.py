# from transformers import BertModel
import torch as t
from config.KBConfig import *
from train.GRUAttnNet2 import GRUAttnNet
from train.GRUNet import LSTMNet, RNNNet
from train.GRURelAttnNet import GRURelAttnNet
from train.HighwayNet import Highway


class RelationModel(t.nn.Module):
    def __init__(self, rel_count1, rel_count2, all_embed1, all_embed2, device):
        super(RelationModel, self).__init__()
        self.device = t.device('cuda')
        # rel_embedding_dim = 64
        self.score_distance_level = SCORE_DISTANCE_LEVEL
        self.rel_count1 = rel_count1
        self.rel_count2 = rel_count2
        self.rel_embedding1 = None
        self.rel_embedding2 = None
        # self.rel_embedding1 = t.nn.Embedding(rel_count1 + 1, rel_embedding_dim, padding_idx=rel_count1)
        # self.rel_embedding2 = t.nn.Embedding(rel_count2 + 1, rel_embedding_dim, padding_idx=rel_count2)
        # t.nn.init.xavier_uniform_(self.rel_embedding1.weight, gain=1.414)
        # t.nn.init.xavier_uniform_(self.rel_embedding2.weight, gain=1.414)
        embedding_dim = bert_output_dim

        # self.rnn = GRUAttnNet(bert_output_dim, bert_output_dim, 1, 0.2, device)

        # rnn_hidden_dim = 64
        rnn_hidden_dim = 64
        # attr_out_dim = 128
        # classify_out_dim = 2
        self.rnn = GRUAttnNet(embedding_dim, rnn_hidden_dim, 1, dropout=0.2, device=device)
        # self.rnn2 = GRURelAttnNet(embedding_dim, rnn_hidden_dim, 1, dropout=0.2, device=device)
        # self.rnn = GRUAttnNet(embedding_dim, rnn_hidden_dim, 1, dropout=0.2, device=device)
        # self.rnn = LSTMNet(embedding_dim, rnn_hidden_dim, 1, dropout=0.2, device=device)
        # self.rnn = RNNNet(embedding_dim, rnn_hidden_dim, 1, dropout=0.2, device=device)
        self.bn = t.nn.Sequential(
            t.nn.BatchNorm1d(rnn_hidden_dim),
            # t.nn.BatchNorm1d(bert_output_dim + rnn_hidden_dim),
            # t.nn.BatchNorm1d(bert_output_dim),
            t.nn.Softsign(),
        ).to(device)
        # #########2021-11-17
        attr_out_dim = 300
        self.combiner = t.nn.Sequential(
            t.nn.Linear(attr_out_dim+rnn_hidden_dim, attr_out_dim),
            t.nn.Dropout(0.15, inplace=True),
            t.nn.BatchNorm1d(attr_out_dim),
            t.nn.ReLU(inplace=True),
            Highway(attr_out_dim, device),
            t.nn.BatchNorm1d(attr_out_dim),
            t.nn.ReLU(),
            # t.nn.Linear(attr_out_dim, attr_out_dim),
        ).to(device)
        # #########2021-11-17
        # self.att = AttrModel(8)
        # self.attr_classifier = t.nn.Sequential(
        #     t.nn.Linear(rnn_hidden_dim * 2, attr_out_dim),
        #     t.nn.Dropout(0.15, inplace=True),
        #     t.nn.BatchNorm1d(attr_out_dim),
        #     t.nn.ReLU(inplace=True),
        #     Highway(attr_out_dim, device),
        #     t.nn.BatchNorm1d(attr_out_dim),
        #     t.nn.ReLU(),
        #     t.nn.Linear(attr_out_dim, attr_out_dim),
        # ).to(device)
        self.ent_embedding1 = t.nn.Embedding.from_pretrained(all_embed1.detach(), padding_idx=all_embed1.shape[0]-1)
        self.ent_embedding2 = t.nn.Embedding.from_pretrained(all_embed2.detach(), padding_idx=all_embed2.shape[0]-1)
        self.to(device)

    def forward(self, pe1s, pe2s, ne1s, ne2s, bpn1s, bpn2s, bnn1s, bnn2s, bpr1s, bpr2s, bnr1s, bnr2s):
        pe1s, pe2s, ne1s, ne2s = pe1s.to(self.device), pe2s.to(self.device), ne1s.to(self.device), ne2s.to(self.device)
        bpn1s, bpn2s, bnn1s, bnn2s = bpn1s.to(self.device), bpn2s.to(self.device), bnn1s.to(self.device), bnn2s.to(self.device)
        bpr1s, bpr2s, bnr1s, bnr2s = bpr1s.to(self.device), bpr2s.to(self.device), bnr1s.to(self.device), bnr2s.to(self.device)
        # pr1s = self.get_rel_embeds(bpn1s, bpr1s, pe1s, self.rel_embedding1, self.ent_embedding1)
        # pr2s = self.get_rel_embeds(bpn2s, bpr2s, pe2s, self.rel_embedding2, self.ent_embedding2)
        # nr1s = self.get_rel_embeds(bnn1s, bnr1s, ne1s, self.rel_embedding1, self.ent_embedding1)
        # nr2s = self.get_rel_embeds(bnn2s, bnr2s, ne2s, self.rel_embedding2, self.ent_embedding2)
        pr1s = self.get_rel_embeds(bpn1s, bpr1s, pe1s, self.rel_embedding1, self.ent_embedding1)
        pr2s = self.get_rel_embeds(bpn2s, bpr2s, pe2s, self.rel_embedding2, self.ent_embedding2)
        nr1s = self.get_rel_embeds(bnn1s, bnr1s, ne1s, self.rel_embedding1, self.ent_embedding1)
        nr2s = self.get_rel_embeds(bnn2s, bnr2s, ne2s, self.rel_embedding2, self.ent_embedding2)
        # #########2021-11-17
        # pos_emb1, pos_emb2 = pr1s, pr2s
        pos_emb1, pos_emb2 = t.cat((pe1s, pr1s), dim=1), t.cat((pe2s, pr2s), dim=1)
        pos_emb1 = self.combiner(pos_emb1)
        pos_emb2 = self.combiner(pos_emb2)
        # #########2021-11-17
        # #########2021-11-18
        pos_emb1 = t.cat((pr1s, pos_emb1), dim=1)
        pos_emb2 = t.cat((pr2s, pos_emb2), dim=1)
        # #########2021-11-18
        pos_score = t.nn.functional.pairwise_distance(pos_emb1, pos_emb2, p=self.score_distance_level, keepdim=True)
        y_pred1 = t.cosine_similarity(pos_emb1, pos_emb2).reshape(pe1s.shape[0], 1)
        y1_0 = t.ones([pe1s.shape[0], 1]).cuda() - y_pred1
        y_pred1 = t.cat([y1_0, y_pred1], dim=1)
        # #########2021-11-17
        # neg_emb1, neg_emb2 = nr1s, nr2s
        neg_emb1, neg_emb2 = t.cat((ne1s, nr1s), dim=1), t.cat((ne2s, nr2s), dim=1)
        neg_emb1 = self.combiner(neg_emb1)
        neg_emb2 = self.combiner(neg_emb2)
        # #########2021-11-17
        # #########2021-11-18
        neg_emb1 = t.cat((nr1s, neg_emb1), dim=1)
        neg_emb2 = t.cat((nr2s, neg_emb2), dim=1)
        # #########2021-11-18
        neg_score = t.nn.functional.pairwise_distance(neg_emb1, neg_emb2, p=self.score_distance_level, keepdim=True)
        y_pred2 = t.cosine_similarity(neg_emb1, neg_emb2).reshape(pe1s.shape[0], 1)
        y2_0 = t.ones([pe1s.shape[0], 1]).cuda() - y_pred2
        y_pred2 = t.cat([y2_0, y_pred2], dim=1)
        y = -t.ones(pos_score.shape).cuda()
        # return pos_score, neg_score, y_pred1, y_pred2, y
        # return pos_score, neg_score, y

        loss = t.nn.MarginRankingLoss(MARGIN)(pos_score, neg_score, y)
        y_pred = t.cat((y_pred1, y_pred2))
        labels = t.cat((t.ones(y_pred1.shape[0], dtype=t.long), t.zeros(y_pred1.shape[0], dtype=t.long))).cuda()
        return y_pred, labels, loss

        # x1 = t.cat((pr1s, pr2s), dim=1)
        # x1 = self.attr_classifier(x1)
        # x2 = t.cat((nr1s, nr2s), dim=1)
        # x2 = self.attr_classifier(x2)
        # y_pred = t.cat((x1, x2), dim=0)
        # labels = t.cat((t.ones(x1.shape[0], dtype=t.long), t.zeros(x2.shape[0], dtype=t.long))).cuda()
        # # print('y_pred.shape', y_pred.shape)
        # # print('labels.shape', labels.shape)
        # # loss = t.nn.CrossEntropyLoss(weight=t.FloatTensor(pos_neg).to(device), reduction='mean')(y_pred, y_true)
        # loss = t.nn.CrossEntropyLoss(reduction='mean')(y_pred, labels)
        # return y_pred, labels, loss
        pass

    @staticmethod
    def pos_neg_count(y_true: t.Tensor, batch_size: int):
        pos_count = y_true.sum()
        pos_count = pos_count.item()
        neg_count = batch_size - pos_count
        return pos_count, neg_count

    def case_study(self, batch, rel_embedding: t.nn.Embedding, all_embed, mode) -> (t.Tensor, t.Tensor):
        with t.no_grad():
            ents, fs = batch
            ents = ents.cuda()
            bns, brs = self.get_neighbors_batch(fs, all_embed.weight.shape[0] - 1)

            pad_idx = all_embed.weight.shape[0] - 1
            batch_neighbors = bns
            ones = t.ones(batch_neighbors.shape).cuda()
            zeros = t.ones(batch_neighbors.shape).cuda()
            neighbor_mask = t.where(batch_neighbors == pad_idx, ones, zeros)

            batch_nei_embs: t.Tensor = all_embed(bns)
            h = batch_nei_embs  # (bc, N, out)
            h_prime, weights = self.rnn(h, neighbor_mask)
            rel_embs = self.bn(h_prime)
            if mode == 'all':
                final_embds = t.cat((ents, rel_embs), dim=1).cuda()
            elif mode == 'rel':
                final_embds = rel_embs
            else:
                print('mode error')
                exit(-1)
            return final_embds, weights


    def get_valid_emb(self, batch, rel_embedding: t.nn.Embedding, all_embed, mode):
        with t.no_grad():
            ents, fs = batch
            ents = ents.cuda()
            # print(rel_embedding.weight.shape[0])
            # if rel_embedding == self.rel_embedding1:
            #     rel_pad_idx = self.rel_count1
            # else:
            #     rel_pad_idx = self.rel_count2
            # print('rel_pad_idx', rel_pad_idx)
            pad_idx = all_embed.weight.shape[0] - 1
            bns, brs = self.get_neighbors_batch(fs, pad_idx)

            rel_embs = self.get_rel_embeds(bns, brs, ents, rel_embedding, all_embed)
            if mode == 'all':
                # final_embds = t.add(ents, 1, rel_embs).cuda()
                # final_embds = (ents + rel_embs).cuda()
                final_embds = t.cat((ents, rel_embs), dim=1).cuda()
                # final_embds = t.nn.functional.normalize(final_embds, p=2, dim=-1)
                # final_embds = ents
            elif mode == 'rel':
                final_embds = rel_embs
            else:
                print('mode error')
                exit(-1)
            return final_embds

    def get_rel_embeds(self, batch_neighbors, batch_relations, batch_ent: t.Tensor, rel_embedding, all_embed: t.nn.Embedding):
        pad_idx = all_embed.weight.shape[0] - 1
        ones = t.ones(batch_neighbors.shape).cuda()
        zeros = t.ones(batch_neighbors.shape).cuda()
        neighbor_mask = t.where(batch_neighbors == pad_idx, ones, zeros)

        # print(batch_neighbors.size())
        batch_nei_embs: t.Tensor = all_embed(batch_neighbors)
        # batch_rel_embs: t.Tensor = rel_embedding(batch_relations)
        # print('batch_nei_embs.shape', batch_nei_embs.shape)
        # print(batch_nei_embs)
        h = batch_nei_embs  # (bc, N, out)
        h = t.nn.functional.relu(h)
        h_prime, _ = self.rnn(h, neighbor_mask)
        h_prime = self.bn(h_prime)
        # h_prime = self.rnn2(h, batch_rel_embs)
        # h_prime = self.bn(h_prime)
        # h_prime = self.att(h, batch_ent)
        # h_prime = t.nn.functional.normalize(h_prime, p=2, dim=-1)
        return h_prime

    @staticmethod
    def get_neighbors_batch(batch_facts, pad_idx, pad_idxr=None):
        lens = [len(facts) if facts is not None else 0 for facts in batch_facts]
        N = max(lens)
        batch_neighbors = [[ent for rel, ent in facts] if facts is not None else [] for facts in batch_facts]
        # batch_relations = [[rel for rel, ent in facts] for facts in batch_facts]
        # for neighbors, relations in zip(batch_neighbors, batch_relations):
        #     while len(neighbors) < N:
        #         neighbors.append(pad_idx)
        #         relations.append(pad_idxr)
        for neighbors in batch_neighbors:
            while len(neighbors) < N:
                neighbors.append(pad_idx)

        batch_neighbors = t.tensor(batch_neighbors, dtype=t.long).cuda()
        # batch_relations = t.tensor(batch_relations, dtype=t.long).cuda()
        # return batch_neighbors, batch_relations
        return batch_neighbors, None

    # @staticmethod
    # def get_neighbors(facts, pad_idx):
    #     lens = [len(facts) for facts in batch_facts]
    #     N = max(lens)
    #     batch_neighbors = [[ent for rel, ent in facts] for facts in batch_facts]
    #     for neighbors in batch_neighbors:
    #         while len(neighbors) < N:
    #             neighbors.append(pad_idx)
    #     batch_neighbors = t.tensor(batch_neighbors, dtype=t.long).cuda()
    #     return batch_neighbors


# class AttLayer(t.nn.Module):
#     def __init__(self, input_dim, embedding_dim):
#         super(AttLayer, self).__init__()
#         self.W = t.nn.Parameter(t.empty(size=(input_dim, embedding_dim)), requires_grad=True)
#         t.nn.init.xavier_uniform_(self.W.data, gain=1.414)
#         # self.WW = t.nn.Sequential(
#         #     t.nn.Linear(input_dim, embedding_dim),
#         #     t.nn.ReLU(),
#         # )
#
#         self.a = t.nn.Parameter(t.empty(size=(bert_output_dim + embedding_dim, 1)), requires_grad=True)
#         t.nn.init.xavier_uniform_(self.a.data, gain=1.414)
#         # self.aa = t.nn.Linear(bert_output_dim + embedding_dim, 1)
#         # self.leakyrelu = t.nn.LeakyReLU(0.2)
#         self.leakyrelu = t.nn.ReLU()
#
#     def forward(self, batch_nei_embs, batch_ent: t.Tensor):
#         # batch_nei_embs: t.Tensor = all_embed(batch_neighbors)
#         h = batch_nei_embs  # (bc, N, out)
#         Wh = t.matmul(h, self.W)    # h.shape: (bc, N, in_features), Wh.shape: (bc, N, out_features)
#         # Wh = self.WW(h)
#         # print(Wh.shape)
#         N = Wh.size()[1]
#         Wh1 = batch_ent.reshape(batch_ent.shape[0], 1, batch_ent.shape[1])
#         # Wh1 = Wh1.repeat(1, N, 1)
#         Wh1 = Wh1.repeat_interleave(N, dim=1)
#         # print('Wh1.shape', Wh1.shape)
#         Wh2 = Wh    # (bc, N, out)
#         # print('Wh2.shape', Wh2.shape)
#         a_input = t.cat([Wh1, Wh2], dim=2)
#         e = self.leakyrelu(t.matmul(a_input, self.a))  # e: (bc, N, 2out) * (bc, 2out, 1): (bc, N, 1)
#         # e = self.aa(a_input)
#         # e = self.leakyrelu(e)
#         # print(e.shape)
#         attention = e.transpose(-2, -1)     # (bc, 1, N)
#         attention = t.nn.functional.softmax(attention, dim=2)
#         attention = t.nn.functional.dropout(attention, 0.6)
#         h_prime = t.matmul(attention, Wh).squeeze(dim=1)  # (bc, 1, N) * (bc, N, out) = (bc, 1, out) : (bc, out)
#         return h_prime
#
#
# class AttrModel(t.nn.Module):
#     def __init__(self, nheads, dropout=0.2):
#         """Dense version of GAT."""
#         super(AttrModel, self).__init__()
#         self.dropout = dropout
#
#         self.attentions = [AttLayer(bert_output_dim, 128) for _ in range(nheads)]
#         for i, attention in enumerate(self.attentions):
#             self.add_module('attention_{}'.format(i), attention)
#
#         self.out_att = AttLayer(128 * nheads, 128)
#
#     def forward(self, batch_nei_embs, batch_ent):
#         x = t.nn.functional.dropout(batch_nei_embs, self.dropout, training=self.training)
#         x = t.cat([att(x, batch_ent) for att in self.attentions], dim=1)
#         x = t.nn.functional.dropout(x, self.dropout, training=self.training)
#         # print(x.shape)
#         x = x.reshape(x.shape[0], 1, x.shape[1])
#         # print(x.shape)
#         x = t.nn.functional.elu(self.out_att(x, batch_ent))
#         return t.nn.functional.log_softmax(x, dim=1)


if __name__ == '__main__':
    # print(t.zeros((3, 4)))
    # a = t.tensor([[1, 2, 3], [3, 4, 5]], dtype=t.float)
    # emb = t.nn.Embedding.from_pretrained(a, freeze=False)
    # b = emb(t.tensor([0, 1, 0, 1], dtype=t.long))
    # print(b)
    # # c = t.nn.functional.softmax(b, dim=0)
    # linear = t.nn.Linear(3, 1)
    # bb = linear(b)
    # print(bb)
    # c = t.softmax(bb, dim=-2).t()
    # print(c)
    # d = t.mm(c, b)
    # print(d)
    # a = t.tensor([[1], [2], [3]])
    # c = t.nn.functional.softmax(a, dim=1)
    pass

