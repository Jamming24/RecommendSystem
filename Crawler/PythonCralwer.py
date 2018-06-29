# coding=utf-8

import os
import random
import time
from multiprocessing import Pool
from multiprocessing import Process
from multiprocessing import Queue


# Python 多进程

# 写数据进程的执行代码
def proc_writer(q, urls):
    print('Process(%s) is writing ...' % os.getpid())
    for url in urls:
        q.put(url)
        print('Put %s to queue...' % url)
        time.sleep(random.random())


# 读数据进程执行的代码
def proc_read(q):
    print('Process(%s) is reading...' % os.getpid())
    while True:
        url = q.get(True)
        print('Get %s from queue.' % url)

if __name__ == '__main__':
    # 父进程创建Queue，并传递给每个子进程
    q = Queue()
    proc_writer1 = Process(target=proc_writer, args=(q, ['url_1', 'url_2', 'url_3']))
    proc_writer2 = Process(target=proc_writer, args=(q, ['url_4', 'url_5', 'url_6']))
    proc_reader = Process(target=proc_read, args=(q, ))
    # 启动子进程pro_writer,写入
    proc_writer1.start()
    proc_writer2.start()
    # 启动子进程的读取
    proc_reader.start()
    # 等待proc_writer结束
    proc_writer1.join()
    proc_writer2.join()

    # 这里是死循环，无法等待其结束，只能强行终止；
    proc_reader.terminate()

# 子进程要执行的代码

def run_proc(name):
    # 进程要执行的代码块
    print('Child process %s (%s) Running...' % (name, os.getpid()))


# if __name__ == '__main__':
#     print('Parent process %s.' % os.getpid())
#     for i in range(5):
#         p = Process(target=run_proc, args=(str(i),))
#         print('Process will start.')
#         p.start()
#     p.join()
#     print('Process end.')

# 进程池样例
def run_task(name):
    print('Task %s (pid = %s) is running...' % (name, os.getpid()))
    time.sleep(random.random() * 3)
    print('Task %s end.' % name)


# if __name__ == '__main__':
#     print('Current process %s.' % os.getpid())
#     p = Pool(processes=3)
#     for i in range(5):
#         p.apply_async(run_task, args=(i,))
#     print('Waiting for all subprocesses done...')
#     p.close()
#     p.join()
#     print('All subprocesses done.')
