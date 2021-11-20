from typing import Iterator

from tools.Announce import Announce
from tools.MultiprocessingTool import MultiprocessingTool


def save_dict_reverse(dict: dict, file: str):
    print(Announce.doing(), 'Save dict reverse:', file)
    with open(file, 'w', encoding='utf-8') as wfile:
        for key, value in dict.items():
            print(value, key, sep='\t', file=wfile)
    print(Announce.done(), 'Finished saving', file)


def load_dict_reverse(file: str):
    print(Announce.doing(), 'Load dict reverse:', file)
    with open(file, 'r', encoding='utf-8') as rfile:
        mt = MultiprocessingTool()
        results = mt.packed_solver(lambda line: line.strip('\n').split('\t')).send_packs(rfile).receive_results()
        results = {value: key for key, value in results}
    print(Announce.done(), file)
    return results


def load_dict(file: str):
    print(Announce.doing(), 'load dict:', file)
    with open(file, 'r', encoding='utf-8') as rfile:
        mt = MultiprocessingTool()
        tups = mt.packed_solver(lambda line: line.strip('\n').split('\t')).send_packs(rfile).receive_results()
    dic = {key: value for key, value in tups}
    print(Announce.done(), file)
    return dic


def save_list(l: Iterator, file: str):
    print(Announce.doing(), 'Save list:', file)
    with open(file, 'w', encoding='utf-8') as wfile:
        for tup in l:
            print(*tup, sep='\t', file=wfile)
    print(Announce.done(), 'Finished saving', file)


def load_list(file: str):
    print(Announce.doing(), 'Load list:', file)
    with open(file, 'r', encoding='utf-8') as rfile:
        mt = MultiprocessingTool()
        l = mt.packed_solver(lambda x: x.strip('\n').split('\t')).send_packs(rfile).receive_results()
    print(Announce.done(), file)
    return l


def save_list_p(l: Iterator, file: str):
    print(Announce.doing(), 'Save python list:', file)
    with open(file, 'w', encoding='utf-8') as wfile:
        for tup in l:
            print(tup, file=wfile)
    print(Announce.done(), 'Finished saving', file)


def load_list_p(file: str):
    print(Announce.doing(), 'Load python list:', file)
    with open(file, 'r', encoding='utf-8') as rfile:
        mt = MultiprocessingTool()
        l = mt.packed_solver(lambda x: eval(x)).send_packs(rfile).receive_results()
    print(Announce.done(), file)
    return l


def save_dict_p(dic: dict, file: str):
    print(Announce.doing(), 'Save python dict:', file)
    with open(file, 'w', encoding='utf-8') as wfile:
        for key, value in dic.items():
            print((key,value), file=wfile)
    print(Announce.done(), 'Finished saving', file)
