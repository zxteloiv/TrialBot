from typing import Optional, Tuple
import pickle
import logging
from datetime import datetime as dt
import redis

from trialbot.data.dataset import Dataset

class PickleDataset(Dataset):
    def __init__(self,
                 filename: str,
                 conn: Optional[Tuple[str, int, int]] = None,
                 prefix: str = None,
                 ignore_missing_data: bool = True,
                 ):
        super().__init__()
        self.filename = filename

        self.r = None
        if conn is not None:
            host, port, db = conn
            pool = redis.ConnectionPool(host=host, port=port, db=db)
            self.r = redis.Redis(connection_pool=pool)

        self.prefix = prefix
        self._data: Optional[list] = None
        self._ignore_missing = ignore_missing_data
        self.logger = logging.getLogger(self.__class__.__name__)

    def _read_data(self, force_reading_with_cache: bool = False):
        if self._data is None and (force_reading_with_cache or self.r is None):
            self.logger.info(f'Start reading file {self.filename} at {dt.now().strftime("%H:%M:%S")}')
            self._data = pickle.load(open(self.filename, 'rb'))
            if self.r is not None:
                for i, x in enumerate(self._data):
                    key = self.prefix + str(i)
                    self.r.set(key, pickle.dumps(x))
                    if i % 1000 == 0:
                        self.logger.info(f'Writing the {i}th example into cache')
            self.logger.info(f'End reading file {self.filename} at {dt.now().strftime("%H:%M:%S")}')

    def __len__(self):
        if self.r is not None:
            length = self.r.get(self.prefix + 'len')
            if length is not None:
                return int(length)

        self._read_data()
        return len(self._data)

    def get_example(self, i):
        if self.r is not None:
            example_ = self.r.get(self.prefix + str(i))
            if example_ is None and self._ignore_missing:
                self.logger.warning(f"Forced to ignore the missing data {i}")
                return None
            example = pickle.loads(example_)
            return example

        self._read_data()
        return self._data[i]







