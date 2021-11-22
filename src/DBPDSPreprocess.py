import os
import random
import shutil
from preprocess import Parser
from tools import FileTools

old_version_path = os.path.abspath('../data/DBP15k')
new_version_path = os.path.abspath('../data')

os.chdir(old_version_path)
datasets = os.listdir('.')
print(datasets)


def load_attr(src, dst):
    tups = Parser.for_file(src, Parser.OEAFileType.ttl_full)
    with open(dst, 'w', encoding='utf-8') as wf:
        for tup in tups:
            print(*tup, sep='\t', file=wf)


for dataset in datasets:
    os.chdir('/'.join((old_version_path, dataset)))
    ds1, ds2 = dataset.split('_')
    # print(os.getcwd())
    # 复制数据集文件
    dataset_new_path = '/'.join((new_version_path, dataset))
    os.mkdir(dataset_new_path)
    load_attr('_'.join((ds1, 'att_triples')), '/'.join((dataset_new_path, 'attr_triples_1')))
    load_attr('_'.join((ds2, 'att_triples')), '/'.join((dataset_new_path, 'attr_triples_2')))
    # shutil.copy('attr_triples_1', dataset_new_path)
    # shutil.copy('attr_triples_2', dataset_new_path)
    shutil.copy('ent_ILLs', '/'.join((dataset_new_path, 'ent_links')))
    shutil.copy('_'.join((ds1, 'rel_triples')), dataset_new_path + '/rel_triples_1')
    shutil.copy('_'.join((ds2, 'rel_triples')), dataset_new_path + '/rel_triples_2')
    # 结束
    # folds = os.listdir('mapping')
    # print(folds)
    ent_links = FileTools.load_list('/'.join((dataset_new_path, 'ent_links')))
    random.seed(11037)
    random.shuffle(ent_links)
    ent_len = len(ent_links)
    train_len = ent_len * 2 // 10
    valid_len = ent_len * 1 // 10
    train_links = ent_links[:train_len]
    valid_links = ent_links[train_len: train_len + valid_len]
    test_links = ent_links[train_len + valid_len:]
    new_fold_path = '/'.join((new_version_path, dataset, '721_5fold', '0'))
    os.makedirs(new_fold_path)
    os.chdir(new_fold_path)

    FileTools.save_list(train_links, '/'.join((new_fold_path, 'train_links')))
    FileTools.save_list(valid_links, '/'.join((new_fold_path, 'valid_links')))
    FileTools.save_list(test_links, '/'.join((new_fold_path, 'test_links')))
