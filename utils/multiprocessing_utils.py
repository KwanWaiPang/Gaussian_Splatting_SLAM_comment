import copy

import torch
import torch.multiprocessing as mp


class FakeQueue: #模拟了 Python 多进程队列的简单类
    def put(self, arg): #这个方法模拟了 Queue 对象的 put 方法，但是它并不真正地把数据放入队列中，而是简单地删除传入的参数。
        del arg

    def get_nowait(self): #个方法模拟了 Queue 对象的 get_nowait 方法，但是它总是抛出一个 mp.queues.Empty 异常，表示队列为空。
        raise mp.queues.Empty

    def qsize(self): #这个方法模拟了 Queue 对象的 qsize 方法，但它总是返回 0，表示队列大小为 0。
        return 0

    def empty(self): #这个方法模拟了 Queue 对象的 empty 方法，但它总是返回 True，表示队列为空。
        return True


def clone_obj(obj):
    clone_obj = copy.deepcopy(obj)
    for attr in clone_obj.__dict__.keys():
        # check if its a property
        if hasattr(clone_obj.__class__, attr) and isinstance(
            getattr(clone_obj.__class__, attr), property
        ):
            continue
        if isinstance(getattr(clone_obj, attr), torch.Tensor):
            setattr(clone_obj, attr, getattr(clone_obj, attr).detach().clone())
    return clone_obj
