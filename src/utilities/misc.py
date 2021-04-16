from collections import OrderedDict
import logging


class Registry:
    def __init__(self): self.dd = OrderedDict()
    def keys(self): return list(self.dd.keys())
    def values(self): return list(self.dd.values())
    def items(self): return self.dd.items()
    def get(self, key, default=None): return self.dd.get(key, default)
    def put(self, key, val):
        # print(self.keys())
        assert(key not in self.dd)
        self.dd[key] = val
    def reg(self, tp):
        self.put(tp.__name__, tp)
        return tp


def setup_logger(logger=None, name='', log_level=logging.INFO): # INFO DEBUG
    if not logger: logger = logging.getLogger(name)
    if logger.hasHandlers(): logger.propagate = 0
    logger.setLevel(log_level)
    ch = logging.StreamHandler()
    formatter = logging.Formatter(
            '%(asctime)s|%(name)s|%(message)s',
            # logging.BASIC_FORMAT,
            "%H:%M:%S")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger
