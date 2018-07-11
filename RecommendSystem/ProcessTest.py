import multiprocessing
import os
import time
from datetime import datetime


def subprocess(number,name):
    # 子进程
    print('这是第%d个子进程' % number)
    pid = os.getpid()  # 得到当前进程号
    print('当前进程号：%s，开始时间：%s' % (pid, datetime.now().isoformat()))
    time.sleep(10)  # 当前进程休眠30秒
    print('当前进程号：%s，结束时间：%s' % (pid, datetime.now().isoformat()))
    return name


def mainprocess():
    # 主进程
    print('这是主进程，进程编号：%d' % os.getpid())
    t_start = datetime.now()

    pool = multiprocessing.Pool()
    for i in range(8):
        pool.apply_async(subprocess, args=(i,"好使吗"))
    pool.close()
    pool.join()
    t_end = datetime.now()
    print('主进程用时：%d毫秒' % (t_end - t_start).microseconds)


if __name__ == '__main__':
    # 主测试函数
    mainprocess()
