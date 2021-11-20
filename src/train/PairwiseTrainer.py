from torch.utils.data import DataLoader, TensorDataset, SequentialSampler
from transformers import BertTokenizer, BertModel, AdamW
import torch.nn.functional as F
from preprocess import Parser
from preprocess.BertDataLoader import BertDataLoader
from preprocess.KBStore import KBStore
from tools.Announce import Announce
from tools.ModelTools import ModelTools
from tools.MultiprocessingTool import MPTool, MultiprocessingTool
from tools.TrainingTools import TrainingTools
from train.BasicBertModel import BasicBertModel
from train.PairwiseDataset import PairwiseDataset
from config.KBConfig import *
import numpy as np
import torch as t
from train.train_utils import cos_sim_mat_generate, batch_topk, hits
VALID = True


class PairwiseTrainer(MultiprocessingTool):
    def __init__(self):
        super(PairwiseTrainer, self).__init__()
        self.get_emb_batch = 512
        self.train_emb_batch = 8
        # self.emb_cos_batch = 512
        self.nearest_sample_num = 128
        self.neg_num = 2
        self.score_distance_level = SCORE_DISTANCE_LEVEL

    def data_prepare(
            self, eid2tids1: dict, eid2tids2: dict,
            fs1: KBStore, fs2: KBStore,
    ):
        self.fs1 = fs1
        self.fs2 = fs2
        tokenizer = BertTokenizer.from_pretrained(args.pretrain_bert_path)
        cls_token, sep_token = tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]'])
        freqs1 = BertDataLoader.load_freq(dataset1)
        freqs2 = BertDataLoader.load_freq(dataset2)
        freqs1 = {key[1]: value for key, value in freqs1.items()}
        freqs2 = {key[1]: value for key, value in freqs2.items()}
        # eid2data1 = self.packed_solver(
        #     self.tids_solver,
        #     cls_token=cls_token,
        #     sep_token=sep_token,
        #     freqs=freqs1,
        # ).send_packs(eid2tids1.items()).receive_results()
        # eid2data1 = [self.tids_solver(item, cls_token=cls_token, sep_token=sep_token, freqs=freqs1) for item in eid2tids1.items()]
        eid2data1 = [self.tids_solver(item, cls_token=cls_token, sep_token=sep_token, freqs=None) for item in eid2tids1.items()]
        self.eid2data1 = {key: value for key, value in eid2data1}

        # eid2data1 [eid: (input_ids, masks)]
        # eid2data2 = self.packed_solver(
        #     self.tids_solver,
        #     cls_token=cls_token,
        #     sep_token=sep_token,
        #     freqs=freqs2,
        # ).send_packs(eid2tids2.items()).receive_results()

        # eid2data2 = [self.tids_solver(item, cls_token=cls_token, sep_token=sep_token, freqs=freqs2) for item in eid2tids2.items()]
        eid2data2 = [self.tids_solver(item, cls_token=cls_token, sep_token=sep_token, freqs=None) for item in eid2tids2.items()]
        self.eid2data2 = {key: value for key, value in eid2data2}
        print(Announce.printMessage(), 'eid2data1 len:', len(self.eid2data1))
        print(Announce.printMessage(), 'eid2data2 len:', len(self.eid2data2))

        if not args.blocking:
            # 读取 train_links, valid_links
            # TODO 加入relation之后，要区分有data（属性）的实体eid2data.keys()和全部实体entity_ids.values()
            self.train_links = self.load_links(links.train, fs1.entity_ids, fs2.entity_ids)
            self.valid_links = self.load_links(links.valid, fs1.entity_ids, fs2.entity_ids)
            self.test_links = self.load_links(links.test, fs1.entity_ids, fs2.entity_ids)
            self.train_links = list(set(self.train_links))
            self.valid_links = list(set(self.valid_links))
            self.test_links = list(set(self.test_links))
            print(Announce.printMessage(), 'train links len:', len(self.train_links))
            print(Announce.printMessage(), 'valid links len:', len(self.valid_links))
            print(Announce.printMessage(), 'test links len:', len(self.test_links))
            self.train_links_p = [(e1, e2) for e1, e2 in self.train_links if e1 in self.eid2data1 and e2 in self.eid2data2]
            # self.train_links_r = [(e1, e2) for e1, e2 in self.train_links if e1 in fs1.facts and e2 in fs2.facts]
            self.valid_links_p = [(e1, e2) for e1, e2 in self.valid_links if e1 in self.eid2data1 and e2 in self.eid2data2]
            self.test_links_p = [(e1, e2) for e1, e2 in self.test_links if e1 in self.eid2data1 and e2 in self.eid2data2]
            self.train_ent1s = list({e1 for e1, e2 in self.train_links})
            self.train_ent2s = list({e2 for e1, e2 in self.train_links})
            self.valid_ent1s = [e1 for e1, e2 in self.valid_links]
            self.valid_ent2s = [e2 for e1, e2 in self.valid_links]
            self.test_ent1s = [e1 for e1, e2 in self.test_links]
            self.test_ent2s = [e2 for e1, e2 in self.test_links]

            self.train_ent1s_p = list({e1 for e1, e2 in self.train_links_p})
            self.train_ent2s_p = list({e2 for e1, e2 in self.train_links_p})
            # self.train_ent1s_r = list({e1 for e1, e2 in self.train_links_r})
            # self.train_ent2s_r = list({e2 for e1, e2 in self.train_links_r})
            self.valid_ent1s_p = [e1 for e1, e2 in self.valid_links_p]
            self.valid_ent2s_p = [e2 for e1, e2 in self.valid_links_p]
            self.test_ent1s_p = [e1 for e1, e2 in self.test_links_p]
            self.test_ent2s_p = [e2 for e1, e2 in self.test_links_p]
            self.all_ent1s_p = list(self.eid2data1.keys())
            self.all_ent2s_p = list(self.eid2data2.keys())
            self.all_ent1s = list(fs1.entity_ids.values())
            self.all_ent2s = list(fs2.entity_ids.values())
            self.all_ent1s_p_idx = [self.all_ent1s.index(ent) for ent in self.all_ent1s_p]
            self.all_ent2s_p_idx = [self.all_ent2s.index(ent) for ent in self.all_ent2s_p]
            # print(self.all_ent1s_p_idx)
            # print(self.all_ent2s_p_idx)
            # self.all_ent1s = self.train_ent1s
            # self.all_ent2s = self.train_ent2s
            '''
            挪到generate_train_tups里面
            
            self.train_ent_idx1s = [self.all_ent1s_p.index(e) for e in self.train_ent1s]
            self.train_ent_idx2s = [self.all_ent2s_p.index(e) for e in self.train_ent2s]
            self.valid_ent_idx1s = [self.all_ent1s_p.index(e) for e in self.valid_ent1s]
            self.valid_ent_idx2s = [self.all_ent2s_p.index(e) for e in self.valid_ent2s]
            '''
            self.block_loader1, self.block_loader2 = self.links_pair_loader(self.all_ent1s_p, self.all_ent2s_p)
            if VALID:
                self.valid_link_loader1, self.valid_link_loader2 = self.links_pair_loader(self.valid_ent1s_p, self.valid_ent2s_p)
            self.test_link_loader1, self.test_link_loader2 = self.links_pair_loader(self.test_ent1s_p, self.test_ent2s_p)
            '''
            获取ILL中的entity
            全部entity，作为初筛的候选实体
            顺序输入模型得到embedding
            获取ILL和all中的embedding
            二重循环计算相似度矩阵
            取top128 后随机选择nagative
            构造(psbj, pobj, nsbj, nobj)
            
            利用(ps, po, ns, no)正常训练
            '''
        pass

    def links_pair_loader(self, ent1s, ent2s):
        inputs1 = self.get_tensor_data(ent1s, self.eid2data1)
        inputs2 = self.get_tensor_data(ent2s, self.eid2data2)
        ds1 = TensorDataset(*inputs1)
        ds2 = TensorDataset(*inputs2)
        sampler1 = SequentialSampler(ds1)
        sampler2 = SequentialSampler(ds2)
        loader1 = DataLoader(ds1, sampler=sampler1, batch_size=self.get_emb_batch)
        loader2 = DataLoader(ds2, sampler=sampler2, batch_size=self.get_emb_batch)
        return loader1, loader2

    @staticmethod
    def get_tensor_data(ents: list, eid2data: dict):
        inputs = [eid2data.get(key) for key in ents]
        print(len(inputs))
        input_ids = [ids for ids, mask in inputs]
        input_ids = t.stack(input_ids, dim=0)
        masks = [mask for ids, mask in inputs]
        masks = t.stack(masks, dim=0)
        return input_ids, masks

    def train(
            self, epochs=100,
            max_grad_norm=1.0,
            learning_rate=2e-5,
            adam_eps=1e-8,
            warmup_steps=0,
            weight_decay=0.0,
            device='cpu',
    ):
        '''
        每一轮
        '''
        # bert_model = BertModel.from_pretrained(args.pretrain_bert_path)
        bert_model = BasicBertModel(args.pretrain_bert_path)
        # bert_model = ModelTools.load_model(args.basic_bert_path)
        # rel_model = RelationModel(len(self.fs1.relation_ids), len(self.fs2.relation_ids), device)
        if PARALLEL:
            bert_model = t.nn.DataParallel(bert_model)
        bert_model.to(device)
        criterion = t.nn.MarginRankingLoss(MARGIN)
        optimizer = AdamW(bert_model.parameters(), lr=1e-5)
        # import transformers
        # transformers.get_linear_schedule_with_warmup(optimizer, warmup_steps, len())
        # optimizer, scheduler = self.build_optimizer(bert_model,
        #                                             num_train_steps,
        #                                             learning_rate,
        #                                             adam_eps,
        #                                             warmup_steps,
        #                                             weight_decay)
        print(optimizer)

        self.get_hits(bert_model, self.test_link_loader1, self.test_link_loader2, device=device)
        # max_hit1 = 0
        if VALID:
            model_tool = ModelTools(3, 'max')
        # max_epochs = 100
        for epoch in range(1, epochs + 1):
            print(Announce.doing(), 'Epoch', epoch, '/', epochs, 'start')
            # train
            train_tups = self.generate_train_tups(bert_model, self.train_ent1s_p, self.train_ent2s_p, self.train_links_p, device)
            print(Announce.printMessage(), 'train_tups.shape:', np.array(train_tups).shape)

            # t.cuda.empty_cache()
            bert_model.train()
            train_ds = PairwiseDataset(train_tups, self.eid2data1, self.eid2data2)
            train_samp = SequentialSampler(train_ds)
            train_loader = DataLoader(train_ds, sampler=train_samp, batch_size=self.train_emb_batch)
            tt = TrainingTools(train_loader, device=device)
            # for i, batch in TrainingTools.batch_iter(train_loader):
            # 先训练实体embedding，然后固定实体embedding，再训练relation embedding
            for i, batch in tt.batches(lambda batch: len(batch[0][0])):
                optimizer.zero_grad()
                loss = self.train_batch(bert_model, tt, batch, criterion, device)

                loss.backward()
                optimizer.step()
                # t.cuda.empty_cache()
            print()
            # TODO relation
            # bert_model.eval()
            # print(Announce.printMessage(), 'Relation')
            # def a(link_loader1, link_loader2, bert_model, device):
            #     with t.no_grad():
            #         all_embed1s_p = self.get_emb_valid(link_loader1, bert_model, device=device)
            #         all_embed2s_p = self.get_emb_valid(link_loader2, bert_model, device=device)
            #         print(Announce.printMessage(), 'all_embed1s_p.shape', all_embed1s_p.shape)
            #         print(Announce.printMessage(), 'all_embed2s_p.shape', all_embed2s_p.shape)
            #         all_embed1s = t.zeros((len(self.all_ent1s), all_embed1s_p.shape[1]), dtype=all_embed1s_p.dtype)
            #         all_embed2s = t.zeros((len(self.all_ent2s), all_embed2s_p.shape[1]), dtype=all_embed2s_p.dtype)
            #         for idx, embed in zip(self.all_ent1s_p, all_embed1s_p):
            #             all_embed1s[idx] = embed
            #         for idx, embed in zip(self.all_ent2s_p, all_embed2s_p):
            #             all_embed2s[idx] = embed
            #         return all_embed1s, all_embed2s
            #
            # all_embed1s, all_embed2s = a(self.block_loader1, self.block_loader2, bert_model, device)
            # # all_embed1s = t.zeros((len(self.all_ent1s), bert_output_dim)).to(device)
            # # all_embed2s = t.zeros((len(self.all_ent1s), bert_output_dim)).to(device)
            # all_embed1s = all_embed1s.to(device)
            # all_embed2s = all_embed2s.to(device)
            # all_embed1s.requires_grad = False
            # all_embed2s.requires_grad = False
            #
            # test_link_loader_r1 = RelationValidDataset(self.test_ent1s, self.fs1, all_embed1s, self.train_emb_batch)
            # test_link_loader_r2 = RelationValidDataset(self.test_ent2s, self.fs2, all_embed2s, self.train_emb_batch)
            # self.get_hits_r(rel_model, test_link_loader_r1, test_link_loader_r2, all_embed1s, all_embed2s, 'rel',
            #                 device=device)
            #
            # train_tups_r = [(pe1, pe2, ne1, ne2) for pe1, pe2, ne1, ne2 in train_tups
            #                 if pe1 in self.fs1.facts and pe2 in self.fs2.facts
            #                 and ne1 in self.fs1.facts and ne2 in self.fs2.facts]
            #
            # print(Announce.printMessage(), 'train_tups len:', len(train_tups))
            # print(Announce.printMessage(), 'train_tups_r len:', len(train_tups_r))
            #
            # rel_train_ds = RelationDataset(train_tups_r, self.fs1, self.fs2, all_embed1s, all_embed2s, self.train_emb_batch)
            # for batch in rel_train_ds:
            #     pos_score, neg_score, y_pred1, y_pred2, y = rel_model(batch, all_embed1s, all_embed2s)
            #     loss = criterion(pos_score, neg_score, y)
            #     del pos_score
            #     del neg_score
            #     del y
            #     if PARALLEL:
            #         loss = loss.mean()
            #     print(float(loss.cpu()), end='')
            #     # batch_size = pos_score.shape[0]
            #     # labels = t.cat([t.ones([batch_size], dtype=t.long), t.zeros([batch_size], dtype=t.long)]).to(device)
            #     # y_pred = t.cat([y_pred1, y_pred2])
            #     # tt.update_metrics(loss, y_pred, labels,
            #     #                   batch_size=batch_size * 2)
            #     loss.backward()
            #     optimizer.step()
            #     # t.cuda.empty_cache()
            # print()
            if VALID:
                print('valid')
                if DEBUG:
                    valid_tups = self.generate_train_tups(bert_model, self.valid_ent1s_p, self.valid_ent2s_p, self.valid_links_p, device)
                    # t.cuda.empty_cache()
                    valid_ds = PairwiseDataset(valid_tups, self.eid2data1, self.eid2data2)
                    valid_samp = SequentialSampler(valid_ds)
                    valid_loader = DataLoader(valid_ds, sampler=valid_samp, batch_size=self.train_emb_batch * 4)
                    vt = TrainingTools(valid_loader, device=device)
                bert_model.eval()
                with t.no_grad():
                    if DEBUG:
                        for i, batch in vt.batches(lambda batch: len(batch[0][0])):
                            loss = self.train_batch(bert_model, vt, batch, criterion, device)
                            # t.cuda.empty_cache()
                    hit_values = self.get_hits(bert_model, self.valid_link_loader1, self.valid_link_loader2, device=device)
                    # if hit_values[0] > max_hit1:
                    #     print("hits@1:", max_hit1, '->', hit_values[0])
                    #     max_hit1 = hit_values
                    stop = model_tool.early_stopping(bert_model, links.model_save, hit_values[0])
            else:
                ModelTools.save_model(bert_model, links.model_save)
            print(Announce.printMessage(), 'test phase')
            self.get_hits(bert_model, self.test_link_loader1, self.test_link_loader2, device=device)
            # test_link_loader_r1 = RelationValidDataset(self.test_ent1s, self.fs1, all_embed1s, self.train_emb_batch)
            # test_link_loader_r2 = RelationValidDataset(self.test_ent2s, self.fs2, all_embed2s, self.train_emb_batch)
            # self.get_hits_r(rel_model, test_link_loader_r1, test_link_loader_r2, all_embed1s, all_embed2s, 'all', device=device)
            print(Announce.done(), 'Epoch', epoch, '/', epochs, 'end')
            if VALID:
                if epoch > 5 and stop:
                    print(Announce.done(), '训练提前结束')
                    stop = False
                    break
            pass
        pass

    def get_hits(self, bert_model: t.nn.Module, link_loader1, link_loader2, device):
        valid_emb1s = self.get_emb_valid(link_loader1, bert_model, device=device)
        valid_emb2s = self.get_emb_valid(link_loader2, bert_model, device=device)
        print(Announce.printMessage(), 'valid_emb1s.shape:', valid_emb1s.shape)
        print(Announce.printMessage(), 'valid_emb2s.shape:', valid_emb2s.shape)
        cos_sim_mat = cos_sim_mat_generate(valid_emb1s, valid_emb2s, device=device)
        _, topk_idx = batch_topk(cos_sim_mat, topn=self.nearest_sample_num, largest=True)
        return hits(topk_idx)

    # def get_hits_r(self, rel_model: RelationModel, link_loader1, link_loader2, all_embed1s, all_embed2s, mode, device):
    #     valid_emb1s = self.get_emb_valid_r(link_loader1, rel_model, rel_model.rel_embedding1, all_embed1s, mode, device=device)
    #     valid_emb2s = self.get_emb_valid_r(link_loader2, rel_model, rel_model.rel_embedding2, all_embed2s, mode, device=device)
    #
    #     print(Announce.printMessage(), 'valid_emb1s.shape:', valid_emb1s.shape)
    #     print(Announce.printMessage(), 'valid_emb2s.shape:', valid_emb2s.shape)
    #     cos_sim_mat = cos_sim_mat_generate(valid_emb1s, valid_emb2s, device=device)
    #     _, topk_idx = batch_topk(cos_sim_mat, topn=self.nearest_sample_num, largest=True)
    #     return hits(topk_idx)

    def train_batch(self, bert_model, tt, batch, criterion, device):
        pos_emb1 = bert_model(batch[0][0].to(device), batch[0][1].to(device))
        pos_emb2 = bert_model(batch[1][0].to(device), batch[1][1].to(device))
        batch_size = pos_emb1.shape[0]
        pos_score = F.pairwise_distance(pos_emb1, pos_emb2, p=self.score_distance_level, keepdim=True)
        # if DEBUG:
        y_pred1 = t.cosine_similarity(pos_emb1, pos_emb2).reshape(batch_size, 1)
        # print(y_pred1.shape)
        y1_0 = t.ones([batch_size, 1]).to(device) - y_pred1
        # print(y1_0.shape)
        y_pred1 = t.cat([y1_0, y_pred1], dim=1)
        # !if DEBUG:
        # del pos_emb1
        # del pos_emb2
        neg_emb1 = bert_model(batch[2][0].to(device), batch[2][1].to(device))
        neg_emb2 = bert_model(batch[3][0].to(device), batch[3][1].to(device))
        neg_score = F.pairwise_distance(neg_emb1, neg_emb2, p=self.score_distance_level, keepdim=True)
        # if DEBUG:
        y_pred2 = t.cosine_similarity(neg_emb1, neg_emb2).reshape(batch_size, 1)
        y2_0 = t.ones([batch_size, 1]).to(device) - y_pred2
        y_pred2 = t.cat([y2_0, y_pred2], dim=1)
        # !if DEBUG:
        # del neg_emb1
        # del neg_emb2
        y = -t.ones(pos_score.shape).to(device)
        loss = criterion(pos_score, neg_score, y)
        # print(pos_score.mean(0).cpu().tolist(), neg_score.mean(0).cpu().tolist(), end='')
        # print()
        # del pos_score
        # del neg_score
        # del y
        if PARALLEL:
            loss = loss.mean()
        print(float(loss), end='')
        # if DEBUG:
        labels = t.cat([t.ones([batch_size], dtype=t.long), t.zeros([batch_size], dtype=t.long)]).to(device)
        y_pred = t.cat([y_pred1, y_pred2])
        tt.update_metrics(loss, y_pred, labels, batch_size=batch_size * 2)
        # !if DEBUG:
        return loss

    def generate_train_tups(self, bert_model, train_ent1s, train_ent2s, train_links, device):
        bert_model.eval()
        all_emb1s = self.get_emb_valid(self.block_loader1, bert_model, device=device)
        all_emb2s = self.get_emb_valid(self.block_loader2, bert_model, device=device)
        train_ent_idx1s = [self.all_ent1s_p.index(e) for e in train_ent1s]
        train_ent_idx2s = [self.all_ent2s_p.index(e) for e in train_ent2s]
        train_emb1s = all_emb1s[train_ent_idx1s]
        print(Announce.printMessage(), 'all_emb1s.shape', all_emb1s.shape)
        print(Announce.printMessage(), 'train_emb1s.shape', train_emb1s.shape)
        # train_emb2s = all_emb1s[train_ent_idx2s]      2021-03-13发现错误，那之前的？？？
        train_emb2s = all_emb2s[train_ent_idx2s]
        print(Announce.printMessage(), 'all_emb2s.shape', all_emb2s.shape)
        print(Announce.printMessage(), 'train_emb2s.shape', train_emb2s.shape)
        # 每个entity生成一个候选实体
        candidate_dic1 = self.get_candidate_dict(train_ent1s, train_emb1s, self.all_ent2s_p, all_emb2s, device=device)
        candidate_dic2 = self.get_candidate_dict(train_ent2s, train_emb2s, self.all_ent1s_p, all_emb1s, device=device)
        train_tups = []
        for pe1, pe2 in train_links:
            for _ in range(self.neg_num):
                if np.random.rand() <= 0.5:
                    # e1
                    # 从50个相似度最高的候选队中挑出一个
                    ne1s = candidate_dic2[pe2]
                    ne1 = ne1s[np.random.randint(self.nearest_sample_num)]
                    ne2 = pe2
                else:
                    ne1 = pe1
                    ne2s = candidate_dic1[pe1]
                    ne2 = ne2s[np.random.randint(self.nearest_sample_num)]
                # same check
                if pe1 != ne1 or pe2 != ne2:
                    # 添加一组positive pair和negative pair
                    train_tups.append([pe1, pe2, ne1, ne2])
            pass
        return train_tups

    def get_candidate_dict(self, train_ents, train_embs, all_ents, all_embs, device):
        cos_sim_mat = cos_sim_mat_generate(train_embs, all_embs, device=device)
        print(Announce.printMessage(), 'cos_sim_mat.shape', cos_sim_mat.shape)
        # print(cos_sim_mat)
        _, topk_idx = batch_topk(cos_sim_mat, topn=self.nearest_sample_num, largest=True)
        print(Announce.printMessage(), 'topk_idx.shape:', topk_idx.shape)
        topk_idx = topk_idx.tolist()
        # topk_idx = [topk[np.random.randint(len(topk))] for topk in topk_idx]
        # candidate_dic = {train_ent: all_ents[all_ent_idx] for train_ent, all_ent_idx in
        #                  zip(train_ents, topk_idx)}
        candidate_dic = {train_ent: [all_ents[all_ent_idx] for all_ent_idx in all_ent_idxs] for train_ent, all_ent_idxs in zip(train_ents, topk_idx)}
        return candidate_dic

    @staticmethod
    def get_emb_valid(loader: DataLoader, model: BertModel or t.nn.DataParallel, device='cpu') -> t.Tensor:
        # results = [model(batch[0].to(device), batch[1].to(device)) for batch in loader]
        results = []
        with t.no_grad():
            model.eval()
            for i, batch in TrainingTools.batch_iter(loader, 'get embedding'):
                # batch = tuple(tup.to(device) for tup in batch)
                emb = model(batch[0].to(device), batch[1].to(device)).cpu()
                # batch = tuple(tup.cpu() for tup in batch)
                results.append(emb)
                # t.cuda.empty_cache()
            embs = t.cat(results, dim=0)
        embs.requires_grad = False
        return embs

    # @staticmethod
    # def get_emb_valid_r(loader: DataLoader, model: RelationModel, rel_embedding, all_embed, mode, device='cpu') -> t.Tensor:
    #     # results = [model(batch[0].to(device), batch[1].to(device)) for batch in loader]
    #     results = []
    #     with t.no_grad():
    #         for batch in loader:
    #             emb = model.get_valid_emb(batch, rel_embedding, all_embed, mode).cpu()
    #             results.append(emb)
    #             # t.cuda.empty_cache()
    #         embs = t.cat(results, dim=0)
    #     return embs

    # def get_emb_train(self, ents, ent2data, device='cpu'):
    #     input_ids, masks = self.get_tensor_data(ents, ent2data)
    #     ds = TensorDataset(input_ids, masks)
    #     sampler = SequentialSampler(ds)
    #     loader = DataLoader(ds, sampler=sampler)
    #     for i, batch in TrainingTools.batch_iter(loader, 'Train get embedding'):
    #         pass

    # def get_candidates(self, bert_model: BertModel):
    #     bert_model.eval()
    #
    #     pass

    @staticmethod
    def load_links(link_path, entity_ids1: dict, entity_ids2: dict):
        def links_line(line: str):
            sbj, obj = Parser.oea_truth_line(line)
            sbj = entity_ids1.get(sbj)
            obj = entity_ids2.get(obj)
            return sbj, obj
        with open(link_path, 'r', encoding='utf-8') as rfile:
            links = MPTool.packed_solver(links_line).send_packs(rfile).receive_results()
            links = list(filter(lambda x: x[0] is not None and x[1] is not None, links))
        return links

    @staticmethod
    def tids_solver(
            item,
            cls_token, sep_token,
            pad_token=0,
            freqs=None,
    ):
        eid, tids = item
        assert eid is not None
        assert len(tids) > 0
        if freqs is None:
            tids = PairwiseTrainer.reduce_tokens(tids, max_len=seq_max_len)
        else:
            tids = PairwiseTrainer.reduce_tokens_with_freq(tids, freqs, max_len=seq_max_len)

        pad_length = seq_max_len - len(tids)
        input_ids = [cls_token] + tids + [pad_token] * pad_length
        masks = [1] * (len(tids) + 1) + [pad_token] * pad_length
        assert len(input_ids) == seq_max_len + 1, len(input_ids)
        assert len(masks) == seq_max_len + 1, len(input_ids)
        # input_ids = np.array(input_ids, dtype=np.long)
        # masks = np.array(masks, dtype=np.long)
        input_ids = t.tensor(input_ids, dtype=t.long)
        masks = t.tensor(masks, dtype=t.long)
        return eid, (input_ids, masks)

    # @staticmethod
    # def data_item_solver(
    #         pair,
    #         eid2tids1, eid2tids2,
    #         cls_token, sep_token,
    #         pad_token=0,
    #         freqs_1=None,
    #         freqs_2=None,
    #         **kwargs
    # ):
    #     label = pair[2]
    #     input_ids1, masks1 = PairwiseTrainer.tids_solver(pair[0], eid2tids1, freqs_1, cls_token, sep_token, pad_token)
    #     input_ids2, masks2 = PairwiseTrainer.tids_solver(pair[1], eid2tids2, freqs_2, cls_token, sep_token, pad_token)
    #     return input_ids1, input_ids2, masks1, masks2, label
    #     pass
    #
    # @staticmethod
    # def tids_solver(
    #         eid, eid2tids, freqs,
    #         cls_token, sep_token,
    #         pad_token=0,
    # ):
    #     tids = eid2tids[eid]
    #     if freqs is None:
    #         tids = Trainer.reduce_tokens(tids)
    #     pad_length = seq_max_len - len(tids)
    #     input_ids = [cls_token] + tids + [pad_token] * pad_length
    #     masks = [1] * (len(tids) + 1) + [pad_token] * pad_length
    #     input_ids = np.array(input_ids, dtype=np.long)
    #     masks = np.array(masks, dtype=np.long)
    #     return input_ids, masks

    @staticmethod
    def reduce_tokens_with_freq(tids, freqs: dict, max_len=200):
        total_length = len(tids)
        if total_length <= max_len:
            return tids
        # 先标注词频和原始顺序
        tids = [(i, token, freqs.get(token)) for i, token in enumerate(tids)]
        # 再按词频大到小排序
        tids = sorted(tids, key=lambda x: x[2], reverse=False)
        while True:
            total_length = len(tids)
            if total_length <= max_len:
                break
            tids.pop()
        # 剔除后最后按原始顺序排序
        tids = sorted(tids, key=lambda x: x[0], reverse=True)
        tids = [token for i, token, freq in tids]
        return tids

    @staticmethod
    def reduce_tokens(tids, max_len=200):
        while True:
            total_length = len(tids)
            if total_length <= max_len:
                break
            tids.pop()
        return tids

    # @staticmethod
    # def build_optimizer(model, num_train_steps, learning_rate, adam_eps, warmup_steps, weight_decay):
    #     # Prepare optimizer and schedule (linear warmup and decay)
    #     no_decay = ['bias', 'LayerNorm.weight']
    #     optimizer_grouped_parameters = [
    #         {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
    #          'weight_decay': weight_decay},
    #         {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    #     ]
    #
    #     optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_eps)
    #
    #     WarmUp(optimizer)
    #     scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=num_train_steps)
    #
    #     return optimizer, scheduler
