import os
import time
from collections import deque
from subprocess import PIPE, Popen

from pyarrow import plasma, serialize, deserialize


class SharedBuffer():

    def __init__(self, path="/tmp/plasma_shared_{}".format(os.getpid()), queue=None, size=2000000000, max_keep=20):
        self._q = queue
        self._path = path
        self._size = size
        self._max_keep = max_keep
        self._kept_ids = deque()
        self._start()
        self._client = plasma.connect(self._path)

    def send(self, raw_data):
        obj_id = self._client.put(raw_data)
        obj_id = obj_id.binary()
        self._kept_ids.append(obj_id)
        to_vanish_ids = []
        while len(self._kept_ids) > self._max_keep:
            to_vanish_ids.append(plasma.ObjectID(self._kept_ids.popleft()))
        self._client.delete(to_vanish_ids)
        return obj_id

    def recv(self):
        new_id = self._q.get()
        try:
            while True:
                newer_id = self._q.get(block=False)
                new_id = newer_id
                time.sleep(0.01)
        except:
            pass
        return self._client.get(plasma.ObjectID(new_id))

    def _start(self):
        try:
            plasma.connect(self._path, num_retries=3)
        except:
            Popen("plasma_store -m {} -s {}".format(self._size, self._path), shell=True, stderr=PIPE)
            time.sleep(0.1)

    def get_path(self):
        return self._path

    def close(self):
        """Close plasma server."""
        os.system("pkill -9 plasma")
