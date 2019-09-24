import platform
import random
import sys
from datetime import datetime
from functools import wraps
from itertools import groupby
from operator import itemgetter
from pathlib import Path
from logging import Formatter, getLogger, DEBUG, StreamHandler
from time import time, perf_counter

import numpy

logger = None


def add_carla_module():
    module_path = Path.home() / 'projects/carla_python/carla/dist/carla-0.9.5-py3.5-linux-x86_64.egg'
    sys.path.append(str(module_path))


class MyFormatter(Formatter):
    width = 50

    def format(self, record):
        width = 50
        datefmt = '%H:%M:%S'
        cpath = '%s:%s:%s' % (record.module, record.funcName, record.lineno)
        cpath = cpath[-width:].ljust(width)
        record.message = record.getMessage()
        s = "[%s - %s] %s" % (self.formatTime(record, datefmt), cpath, record.getMessage())
        if record.exc_info:
            # Cache the traceback text to avoid converting it multiple times
            # (it's constant anyway)
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            if s[-1:] != "\n":
                s = s + "\n"
            s = s + record.exc_text
        return s


def get_logger(name):
    global logger
    if logger is None:
        LEVEL = DEBUG
        logger = getLogger(name)
        logger.setLevel(LEVEL)
        ch = StreamHandler()
        ch.setLevel(LEVEL)
        formatter = MyFormatter()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


def get_timestamp() -> int:
    return int(time() * 1e6)


def datetime_from_timestamp(ts: int) -> datetime:
    return datetime.fromtimestamp(ts / 1e6)


def unique_with_islices(iterable, key: str = None, item_index: int = -1):
    indices = []
    index = 0
    for k, g in groupby(iterable, key):
        l = len(list(g))
        indices.append((k, (index, index + l)))
        index += l
    if item_index >= 0:
        indices = list(map(itemgetter(item_index), indices))
    return indices


def set_random_seed(random_seed: int):
    import torch
    random.seed(random_seed)
    numpy.random.seed(random_seed)
    torch.manual_seed(random_seed)


def timethis(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = perf_counter()
        r = func(*args, **kwargs)
        end = perf_counter()
        print('{}.{} : {:6.2f} seconds'.format(func.__module__, func.__name__, end - start))
        return r
    return wrapper


def fetch_node_name() -> str:
    return platform.node().lower()
