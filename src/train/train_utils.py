import torch
import torch.nn.functional as F
import numpy as np

from tools.Announce import Announce
from tools.MultiprocessingTool import MPTool


def cos_sim_mat_generate(emb1, emb2, device, bs=128):
    """
    return cosine similarity matrix of embedding1(emb1) and embedding2(emb2)
    emb: [batch, entity id, embedding vector]
    """
    emb1.to(device)
    emb2.to(device)
    array_emb1 = F.normalize(torch.FloatTensor(emb1), p=2, dim=1)
    array_emb2 = F.normalize(torch.FloatTensor(emb2), p=2, dim=1)
    res_mat = batch_mat_mm(array_emb1, array_emb2.t(), device, bs=bs)
    return res_mat


def batch_mat_mm(mat1, mat2, device, bs=128):
    # be equal to matmul, Speed up computing with GPU
    res_mat = []
    axis_0 = mat1.shape[0]
    # mat2.to(device)
    for i in range(0, axis_0, bs):
        temp_div_mat_1 = mat1[i:min(i + bs, axis_0)]
        res = temp_div_mat_1.mm(mat2)
        # res_mat.append(res.cpu())
        res_mat.append(res)
    res_mat = torch.cat(res_mat, 0)
    return res_mat


def batch_topk(mat, bs=128, topn=50, largest=False):
    # be equal to topk, Speed up computing with GPU
    res_score = []
    res_index = []
    axis_0 = mat.shape[0]
    for i in range(0, axis_0, bs):
        temp_div_mat = mat[i:min(i + bs, axis_0)]
        score_mat, index_mat = temp_div_mat.topk(topn, largest=largest)
        res_score.append(score_mat.cpu())
        res_index.append(index_mat.cpu())
    res_score = torch.cat(res_score, 0)
    res_index = torch.cat(res_index, 0)
    return res_score, res_index


hits_list = [1, 5, 10, 50]


def hits(index_mat):
    ent1_num, cands_num = index_mat.shape
    print(Announce.printMessage(), 'index_mat.shape', index_mat.shape)
    assert cands_num >= hits_list[-1]
    result_mat = [[1 if index_mat[i][j] == i else 0 for j in range(cands_num)] for i in range(ent1_num)]
    mrr_mat = [sum([1 / (j + 1) if index_mat[i][j] == i else 0 for j in range(cands_num)]) for i in range(ent1_num)]
    # result_mat = MPTool.packed_solver(lambda i: [1 if index_mat[i][j] == i else 0 for j in range(cands_num)]).send_packs(range(ent1_num)).receive_results()
    result_title_str = ""
    result_str = ""
    hit_values = []
    for hits_num in hits_list:
        total_hit = sum([sum(ent_list[:hits_num]) for ent_list in result_mat])
        # hit_list = MPTool.packed_solver(lambda ent_list: sum(ent_list[:hits_num])).send_packs(result_mat).receive_results()
        # total_hit = sum(hit_list)
        hit_value = total_hit / ent1_num
        result_title_str += ''.join(('Hits@', str(hits_num), '\t'))
        result_str += ''.join((str(hit_value), '\t'))
        hit_values.append(hit_value)
    mrr_value = sum(mrr_mat) / len(mrr_mat)
    result_title_str += 'MRR'
    result_str += str(mrr_value)
    print(result_title_str.strip())
    print(result_str.strip())
    return hit_values