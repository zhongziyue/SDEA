import io
import os
import sys
import time

time_str = time.strftime("%Y-%m-%d-%H%M%S", time.localtime())


def make_print_to_file(path='./logs/', name=time_str + '.txt'):
    '''
     example:
    use make_print_to_file() , and the all the information of funtion print , will be write in to a log file
    :param name:
    :param path: the path to save print information
    :return:
    '''

    class Logger(object):
        def __init__(self, filename="Default.log", path="./", terminal=sys.stdout):
            # sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
            self.terminal = terminal
            if not os.path.exists(path):
                os.makedirs(path)
            self.log = open(os.path.join(path, filename), "a", encoding='utf8')

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            # self.terminal.flush()
            self.log.flush()

        def flush(self):
            pass

    sys.stdout = Logger(name, path=path, terminal=sys.stdout)
    sys.stderr = Logger(name, path=path, terminal=sys.stderr)


if __name__ == '__main__':
    make_print_to_file()
    print('Logger Hello World')
