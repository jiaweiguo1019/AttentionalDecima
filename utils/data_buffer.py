import os
import time
import multiprocessing as mp
from subprocess import PIPE, Popen


from pyarrow import plasma, serialize, deserialize


class DataBuffer():

    def __init__(self, queue, path="/tmp/plasma_data_{}".format(os.getpid()), size=2000000000):
        self._q = queue
        self._path = path
        self._size = size
        self._start()
        self._client = plasma.connect(self._path, num_retries=3)

    def send(self, data):
        obj_id = self._client.put(data)
        obj_id = obj_id.binary()
        self._q.put(obj_id)

    def recv(self):
        obj_id = self._q.get()
        obj_id = plasma.ObjectID(obj_id)
        data = self._client.get(obj_id)
        self._client.delete([obj_id])
        return data

    def _start(self):
        try:
            plasma.connect(self._path, num_retries=3)
        except:
            Popen("plasma_store -m {} -s {}".format(self._size , self._path), shell=True, stderr=PIPE)
            time.sleep(0.1)

    def get_path(self):
        return self._path

    def close(self):
        """Close plasma server."""
        os.system("pkill -9 plasma")



