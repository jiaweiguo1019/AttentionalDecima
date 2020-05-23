"""
Graph Convolutional Network
with
Multi-Head Attention Mechanism

Propergate node features among neighbors
via parameterized message passing scheme
"""


import numpy as np
import tensorflow as tf
from tf_op import glorot, ones, zeros
from param import *


class GraphCNN(object):

    def __init__(self, inputs, input_dim, hid_dims, output_dim,
                 max_depth, act_fn, scope='gcn'):

        self.inputs = inputs

        self.input_dim = input_dim
        self.hid_dims = hid_dims
        self.output_dim = output_dim

        self.max_depth = max_depth

        self.act_fn = act_fn
        self.scope = scope

        # message passing
        self.adj_mats = [tf.sparse_placeholder(
            tf.float32, [None, None]) for _ in range(self.max_depth)]
        self.masks = [tf.placeholder(
            tf.float32, [None, 1]) for _ in range(self.max_depth)]

        # initialize message passing transformation parameters
        # h: x -> x'
        self.prep_weights, self.prep_bias = \
            self.init(self.input_dim, self.hid_dims, self.output_dim)

        # initialize attention vector weights and attention biases
        self.att_weights, self.att_biases = self.att_init()

        # graph message passing
        self.outputs = self.forward()

    def att_init(self):
        # Initialize the parameters for graph attention mechanism
        # Initialize num_heads times
        # these weights may need to be re-used
        # e.g., we may want to propagate information multiple times
        # but using the same way of processing the nodes

        weights = []
        biases = []

        for _ in range(args.num_heads):
            att_weight_1 = glorot([1, self.output_dim, 1],  scope=self.scope)
            att_weight_2 = glorot([1, self.output_dim, 1],  scope=self.scope)
            weights.append([att_weight_1, att_weight_2])
            biases.append(zeros([self.output_dim], scope=self.scope))
        
        return weights, biases

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

    def forward(self):
        # message passing among nodes
        # the information is flowing from leaves to roots
        x = self.inputs
        x = tf.expand_dims(x, axis=0)
        num_nodes = tf.shape(x)[1]

        # raise x into higher dimension
        for i in range(len(self.prep_weights)):
            x = tf.matmul(x, self.prep_weights[i])
            x += self.prep_bias[i]
            x = self.act_fn(x)

        out = []

        for i in range(args.num_heads):
            y = x
            for d in range(self.max_depth):
                f_1 = tf.nn.conv1d(y, self.att_weights[i][0], stride=1, padding='VALID')
                f_2 = tf.nn.conv1d(y, self.att_weights[i][1], stride=1, padding='VALID')

                f_1 = tf.reshape(f_1, (num_nodes, 1))
                f_2 = tf.reshape(f_2, (num_nodes, 1))

                f_1 = self.adj_mats[d] * f_1
                f_2 = self.adj_mats[d] * tf.transpose(f_2, [1, 0])

                logits = tf.sparse_add(f_1, f_2)
                lrelu = tf.SparseTensor(
                    indices=logits.indices, 
                    values=self.act_fn(logits.values), 
                    dense_shape=logits.dense_shape)
            
                coefs = tf.sparse_softmax(lrelu)
                # coefs = tf.sparse_reshape(coefs, [nb_nodes, nb_nodes])

                y_fts = tf.squeeze(y)
                vals = tf.sparse_tensor_dense_matmul(coefs, y_fts)
                vals += self.att_biases[i]
                vals = self.act_fn(vals)
                vals = vals * self.masks[d]
                vals = tf.expand_dims(vals, axis=0)
                y = y + vals
            out.append(y)
        
        return  tf.squeeze(tf.add_n(out) / args.num_heads)
