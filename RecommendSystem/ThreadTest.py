# encoding:utf-8

import threading
from time import ctime
import time
from concurrent.futures import ThreadPoolExecutor


# 对于计算密集型 首选使用ProcessPoolExecutors模块
# 对于IO密集型 首选使用ThreadPoolExecutor

def return_future(msg,msn):
    print(msg, ctime())
    time.sleep(10)
    print(msn, '执行完成')
    return msg


# 创建一个线程池
pool = ThreadPoolExecutor(max_workers=2)

# 在线程池中加入两个任务
list = ['hello', 'world', '我是第三个线程', '我是第四个线程', '我是第五个线程']
for i in list:
    f1 = pool.submit(return_future, i,i)
#    print(f1.result())
# f2 = pool.submit(return_future, )
# f3 = pool.submit(return_future, '我是第三个线程')
#     print(f1.done())
pool.shutdown()
# print(f2.result())
# print(f3.result())
# print(f2.done())

# 新版Python 创建线程采用threading 一种新型，方法更多的线程包
# 创建线程的三种方法

# 第一种 创建Thread实例，传给他一个参数

loops = [4, 2]


def loop(nloop, nsec):
    print("start loop", nloop, "at:", ctime())
    sleep(nsec)
    print("loop", nloop, "done at:", ctime())


def main():
    print('Starting at:', ctime())
    threads = []
    nloops = range(len(loops))
    for i in nloops:
        t = threading.Thread(target=loop, args=(i, loops[i]))
        threads.append(t)
    for i in nloops:
        # 开启线程
        threads[i].start()
    for i in nloops:
        # 等待所有线程结束
        threads[i].join()
    print("all Done at:", ctime())


# 第二种 创建Thread的实例，传给它一个可调用的类实例

class ThreadFunc(object):
    def __init__(self, func, args, name=''):
        self.name = name
        self.func = func
        self.args = args

    def __call__(self):
        self.func(*self.args)


def main_2():
    print("Starting at:", ctime())
    threads = []
    nloops = range(len(loops))
    # 创建所有线程
    for i in nloops:
        t = threading.Thread(target=ThreadFunc(loop, (i, loops[i]), loop.__name__))
        threads.append(t)

    for i in nloops:
        # 开启线程
        threads[i].start()
    for i in nloops:
        # 等待所有线程结束
        threads[i].join()
    print("all Done at:", ctime())


# 派生Thread的子类，并创建子类的实例

class MyThread(threading.Thread):
    def __init__(self, func, args, name=''):
        threading.Thread.__init__(self)
        self.name = name
        self.func = func
        self.args = args

    def run(self):
        self.func(*self.args)


def main_3():
    print("Starting at:", ctime())
    threads = []
    nloops = range(len(loops))
    # 创建所有线程
    for i in nloops:
        t = MyThread(loop, (i, loops[i]), loop.__name__)
        threads.append(t)

    for i in nloops:
        # 开启线程
        threads[i].start()
    for i in nloops:
        # 等待所有线程结束
        threads[i].join()
    print("all Done at:", ctime())

# if __name__ == '__main__':
#     main_3()
