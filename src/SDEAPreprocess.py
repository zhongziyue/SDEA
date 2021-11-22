from preprocess.BertDataLoader import BertDataLoader
from preprocess.KBStore import KBStore
from config.KBConfig import *

if __name__ == '__main__':
    fs1 = KBStore(dataset1)
    fs2 = KBStore(dataset2)
    fs1.load_kb()
    fs2.load_kb()

    KBStore.save_blocks(fs1, fs2)

    dl1 = BertDataLoader(dataset1)
    dl2 = BertDataLoader(dataset2)
    dl1.run()
    dl2.run()
