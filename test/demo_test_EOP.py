from src.Frame.EOP import EOP
import time as Ti


def demo1():
    ep = EOP().load()
    begin = Ti.time()
    for i in range(10000):
        ep.interp(55185)
    print(ep.interp(55185))
    print('lagin4: Cost time: %s ms' % ((Ti.time() - begin) * 1000))
    return True

if __name__ == '__main__':
    demo1()