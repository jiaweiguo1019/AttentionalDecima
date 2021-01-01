"""
Graph Summarization Network
with a
LSTM Aggregator

Summarize node features globally
via parameterized aggregation scheme
"""


import numpy as np
from tf_compat import tf
from tf_op import glorot, ones, zeros


class GraphSNN(object):
    def __init__(self, inputs, input_dim, hid_dims, output_dim, act_fn, scope='gsn'):
        # transform the node features into higher-level features
        # then add all the higher-level features of unfinished nodes of each DAG
        # to form a DAG level summarization
        # apply a random permutation to every DAG level summarization
        # then use a LSTM aggregator to form a global level summarization

        self.inputs = inputs

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hid_dims = hid_dims

        self.act_fn = act_fn
        self.scope = scope

        # DAG level and global level summarization
        self.summ_levels = 2

        # graph summarization, hierarchical structure
        self.summ_mats = [tf.sparse_placeholder(
            tf.float32, [None, None]) for _ in range(self.summ_levels)]

        # initialize summarization parameters for each hierarchy
        self.dag_weights, self.dag_bias = \
            self.init(self.input_dim, self.hid_dims, self.output_dim)

#
        # self.global_weights, self.global_bias = \
            # self.init(self.output_dim, self.hid_dims, self.output_dim)
#

        self.cell = tf.contrib.rnn.BasicLSTMCell(self.output_dim)
#
        # self.cell = tf.nn.rnn_cell.GRUCell(self.output_dim)
        # self.cell = tf.contrib.cudnn_rnn.CudnnLSTM(1, self.output_dim)
#

        # graph summarization operation
        self.summaries = self.summarize()

    def init(self, input_dim, hid_dims, output_dim):
        # Initialize the parameters
        # these weights may need to be re-used
        # e.g., we may want to propagate information multiple times
        # but using the same way of processing the nodes
        weights = []
        bias = []

        curr_in_dim = input_dim

        # hidden layers
        for hid_dim in hid_dims:
            weights.append(
                glorot([curr_in_dim, hid_dim], scope=self.scope))
            bias.append(
                zeros([hid_dim], scope=self.scope))
            curr_in_dim = hid_dim

        # output layer
        weights.append(glorot([curr_in_dim, output_dim], scope=self.scope))
        bias.append(zeros([output_dim], scope=self.scope))

        return weights, bias

    def summarize(self):
        # summarize information in each hierarchy
        # e.g., first level summarize each individual DAG
        # second level globally summarize all DAGs
        x = self.inputs

        summaries = []

        # DAG level summary
        s = x
        for i in range(len(self.dag_weights)):
            s = tf.matmul(s, self.dag_weights[i])
            s += self.dag_bias[i]
            s = self.act_fn(s)

        s = tf.sparse_tensor_dense_matmul(self.summ_mats[0], s)
        summaries.append(s)

        # global level summary
        batch_size = tf.shape(self.summ_mats[1])[0]
        s = tf.reshape(s, [batch_size, -1, self.output_dim])
        s = tf.transpose(s, [1, 0, 2])
        s = tf.gather(s, tf.random_shuffle(tf.range(tf.shape(s)[0])))
        s = tf.transpose(s, [1, 0, 2])

        initial_state = self.cell.zero_state(batch_size, tf.float32)
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
                self.cell, s, initial_state=initial_state,
                dtype=tf.float32, time_major=False)
        s = tf.transpose(rnn_outputs, [1, 0, 2])[-1]
#
        # used when the sequence length is not identical
        # batch_size = tf.shape(rnn_outputs)[0]
        # max_len = tf.shape(rnn_outputs)[1]
        # out_size = int(rnn_outputs.get_shape()[2])
        # index = tf.range(0, batch_size) * max_len + (max_len - 1)
        # flat = tf.reshape(rnn_outputs, [-1, out_size])
        # s = tf.gather(flat, index)
#
#
        # a simpler way to obtain a global level summarization
        # for i in range(len(self.global_weights)):
            # s = tf.matmul(s, self.global_weights[i])
            # s += self.global_bias[i]
            # s = self.act_fn(s)

        # s = tf.sparse_tensor_dense_matmul(self.summ_mats[1], s)
#
        summaries.append(s)

        return summaries
