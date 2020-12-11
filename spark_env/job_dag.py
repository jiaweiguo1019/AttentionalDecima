import networkx as nx
import numpy as np
from collections import OrderedDict
from utils import OrderedSet
from param import *
from spark_env.node import NodeDuration

class JobDAG(object):
    def __init__(self, nodes, adj_mat, name):
        # nodes: list of N nodes
        # adj_mat: N by N 0-1 adjacency matrix, e_ij = 1 -> edge from i to j
        assert len(nodes) == adj_mat.shape[0]
        assert adj_mat.shape[0] == adj_mat.shape[1]

        self.name = name

        self.nodes = nodes
        self.adj_mat = adj_mat

        self.num_nodes = len(self.nodes)
        self.num_nodes_done = 0

        # the computation graph needs to be a DAG
        assert is_dag(self.num_nodes, self.adj_mat)

        # get the set of schedule nodes
        self.frontier_nodes = OrderedSet()
        for node in self.nodes:
            if node.is_schedulable():
                self.frontier_nodes.add(node)

        # set of executors currently running on the job
        self.executors = OrderedSet()

        # assign job dag to node
        self.assign_job_dag_to_node()

        # dag is arrived
        self.arrived = False

        # dag is completed
        self.completed = False

        # dag start ime
        self.start_time = None

        # dag completion time
        self.completion_time = np.inf

        # map a executor number to an interval
        self.executor_interval_map = self.get_executor_interval_map()

    def assign_job_dag_to_node(self):
        for node in self.nodes:
            node.job_dag = self

    def get_executor_interval_map(self):
        executor_interval_map = {}
        entry_pt = 0

        # get the left most map
        for e in range(args.executor_data_point[0] + 1):
            executor_interval_map[e] = (args.executor_data_point[0], args.executor_data_point[0])

        # get the center map
        for i in range(len(args.executor_data_point) - 1):
            for e in range(args.executor_data_point[i] + 1, args.executor_data_point[i + 1]):
                executor_interval_map[e] = (args.executor_data_point[i], args.executor_data_point[i + 1])
            # at the data point
            e = args.executor_data_point[i + 1]
            executor_interval_map[e] = (args.executor_data_point[i + 1], args.executor_data_point[i + 1])

        # get the residual map
        if args.exec_cap > args.executor_data_point[-1]:
            for e in range(args.executor_data_point[-1] + 1, args.exec_cap + 1):
                executor_interval_map[e] = (args.executor_data_point[-1], args.executor_data_point[-1])

        return executor_interval_map

    def reset(self):
        for node in self.nodes:
            node.reset()
        self.num_nodes_done = 0
        self.executors = OrderedSet()
        self.frontier_nodes = OrderedSet()
        for node in self.nodes:
            if node.is_schedulable():
                self.frontier_nodes.add(node)
        self.arrived = False
        self.completed = False
        self.completion_time = np.inf

    def update_frontier_nodes(self, node):
        frontier_nodes_changed = False
        for child in node.child_nodes:
            if child.is_schedulable():
                if child.idx not in self.frontier_nodes:
                    self.frontier_nodes.add(child)
                    frontier_nodes_changed = True
        return frontier_nodes_changed

#    def get_nodes_duration(self):
#        # Warning: this is slow O(num_nodes * num_tasks)
#        # get the duration over all nodes
#        duration = 0
#        for node in self.nodes:
#            duration += node.get_node_duration()
#        return duration


class JobDAGDuration(object):
    # A light-weighted extra storage for job_dag duration

    def __init__(self, job_dag):
        self.job_dag = job_dag

        self.node_durations = {node: NodeDuration(node) for node in self.job_dag.nodes}

        for node in self.job_dag.nodes:
            # initialize descendant nodes duration
            self.node_durations[node].descendant_work = \
                np.sum([self.node_durations[n].duration \
                        for n in node.descendant_nodes])
            # initialize descendant nodes task duration
            self.node_durations[node].descendant_cp = \
                np.sum([n.tasks[0].duration \
                        for n in node.descendant_nodes])

        self.job_dag_duration = \
            np.sum([self.node_durations[node].duration \
                    for node in self.job_dag.nodes])

        self.nodes_done = {}

    def update_duration(self):
        work_done = 0
        for node in self.job_dag.nodes:
            if node not in self.nodes_done and node.tasks_all_done:
                work_done += self.node_durations[node].duration
                self.nodes_done[node] = node
        self.job_dag_duration -= work_done


def is_dag(num_nodes, adj_mat):
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_mat[i, j] == 1:
                G.add_edge(i, j)
    return nx.is_directed_acyclic_graph(G)
