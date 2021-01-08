import multiprocessing as mp
import os
import time
from collections import defaultdict

from attentional_decima.algorithm import AttentionalDecima
from utils.data_buffer import DataBuffer
from utils.shared_buffer import SharedBuffer

from params import args


class Learner():

    def __init__(self, recv_actor_q, data_path, shared_path):
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        self._svae_interval = args.svae_interval
        self._data_buf = DataBuffer(recv_actor_q, data_path)
        self._shared_buf = SharedBuffer(shared_path)
        self._alg = AttentionalDecima()

    def train(self, traj, ep, ent_weight):
        self._alg.train(traj, ent_weight)
        print('train-{} done'.format(ep))
        if ep % self._svae_interval == 0:
            self._save_alg()

    def _save_alg(self):
        pass

    def get_vars(self):
        return self._alg.get_vars()
