import os
import random
import shutil

from tools import FileTools

old_version_path = os.path.abspath('../data/entity-alignment-full-data')
new_version_path = os.path.abspath('../data/')

os.chdir(old_version_path)
datasets = os.listdir('.')
print(datasets)

for dataset in datasets:
    if not dataset.endswith('V1'):
        continue
    os.chdir('/'.join((old_version_path, dataset)))

    dataset_new_path = '/'.join((new_version_path, dataset))
    os.mkdir(dataset_new_path)
    shutil.copy('attr_triples_1', dataset_new_path)
    shutil.copy('attr_triples_2', dataset_new_path)
    shutil.copy('ent_links', dataset_new_path)
    shutil.copy('triples_1', dataset_new_path + '/rel_triples_1')
    shutil.copy('triples_2', dataset_new_path + '/rel_triples_2')

    ent_links = FileTools.load_list('ent_links')
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

