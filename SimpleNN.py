import copy
from collections import Iterable

__author__ = 'Jian Xun'


class Neural:
    # a map from input id to input value
    __inputs = None
    # current output
    __output = None
    # a map from input id to input weight
    __weights = None
    __learn_rate = None
    __theta = None
    __default_weight = None

    def __init__(self, learn_rate, ids, default_weight):
        self.__inputs = {}
        self.__output = None
        self.__weights = {}
        self.__learn_rate = learn_rate
        self.__theta = default_weight
        self.__default_weight = default_weight
        for id in ids:
            self.__inputs[id] = None
            self.__weights[id] = default_weight

    def add_link(self, id):
        self.__inputs[id] = None
        self.__weights[id] = self.__default_weight

    def receive(self, inputs):
        for id in self.__inputs:
            self.__inputs[id] = inputs[id]

    def output(self):
        a = self.__theta
        for id in self.__inputs:
            a += self.__weights[id] * self.__inputs[id]
        self.__output = a if a > 0 else 0
        return self.__output

    def back_propagate(self, target):
        deri = 0 if self.__output == 0 else 1
        error = deri * (target - self.__output)
        derror = self.__learn_rate * error
        propagation = {}
        for id in self.__inputs:
            propagation[id] = self.__weights[id] * error
            self.__weights[id] += self.__inputs[id] * derror
        self.__theta += derror
        return propagation


class Configuration:
    __learn_rate = None
    __default_weight = None
    __input_nodes = None
    # records the input id of each node
    # __relations[a] = [b, c] means a receives inputs from b and c
    __relations = None
    __output_nodes = None

    def __init__(self):
        self.__learn_rate = 0.01
        self.__default_weight = 0.5
        self.__input_nodes = []
        self.__relations = {}
        self.__output_nodes = []

    def set_learn_rate(self, learn_rate):
        self.__learn_rate = learn_rate

    def set_default_weight(self, weight):
        self.__default_weight = weight

    def add_input_node(self, ids):
        if isinstance(ids, Iterable):
            for id in ids:
                self.__input_nodes.append(str(id))
        if isinstance(ids, str):
            self.__input_nodes.append(ids)

    def add_output_node(self, ids):
        if isinstance(ids, Iterable):
            for id in ids:
                self.__output_nodes.append(str(id))
        if isinstance(ids, str):
            self.__output_nodes.append(ids)

    def add_relation(self, from_id, to_id):
        from_id = str(from_id)
        to_id = str(to_id)
        if from_id in self.__relations and to_id in self.__relations[from_id]:
            raise LoopException('loop relation found between ' + from_id + ' and ' + to_id)
        if to_id not in self.__relations:
            self.__relations[to_id] = []
        if from_id not in self.__relations[to_id]:
            self.__relations[to_id].append(from_id)

    def get_learn_rate(self):
        return self.__learn_rate

    def get_default_weight(self):
        return self.__default_weight

    def get_input_nodes(self):
        return self.__input_nodes

    def get_output_nodes(self):
        return self.__output_nodes

    def get_inputs(self, id):
        return self.__relations[id]

    def get_relations(self):
        return copy.deepcopy(self.__relations)


class Network:
    __nodes = None
    __input_nodes = None
    __output_nodes = None
    # stores the order of nodes to calculate result
    __topology = None

    def __init__(self, configuration):
        self.__input_nodes = []
        self.__input_nodes.extend(configuration.get_input_nodes())
        self.__output_nodes = []
        self.__output_nodes.extend(configuration.get_output_nodes())
        self.__nodes = {}
        relations = configuration.get_relations()
        for id in relations:
            self.__nodes[id] = Neural(configuration.get_learn_rate(), configuration.get_inputs(id),
                                      configuration.get_default_weight())
        self.__topology = []
        temp_output = []
        for id in self.__input_nodes:
            temp_output.append(id)
        while len(self.__topology) < len(relations):
            for nid in relations:
                if nid not in temp_output:
                    while len(relations[nid]) > 0 and relations[nid][0] in temp_output:
                        relations[nid].pop(0)
                    if len(relations[nid]) == 0:
                        self.__topology.append(nid)
                        temp_output.append(nid)

    def train(self, inputs, expect_outputs):
        for id in self.__input_nodes:
            if id not in inputs:
                raise MissException('input node not found : ' + id)
        for id in self.__output_nodes:
            if id not in expect_outputs:
                raise MissException('output node not found : ' + id)
        self.predict(inputs)
        # then do back propagation
        internal_propagate = {}
        for id in expect_outputs:
            internal_propagate[id] = expect_outputs[id]
        for idx in xrange(len(self.__topology) - 1, -1, -1):
            id = self.__topology[idx]
            temp = self.__nodes[id].back_propagate(internal_propagate[id])
            for tid in temp:
                if tid in internal_propagate:
                    internal_propagate[tid] += temp[tid]
                else:
                    internal_propagate[tid] = temp[tid]

    def predict(self, inputs):
        internal_result = {}
        for id in inputs:
            internal_result[id] = inputs[id]
        # calculate output
        for id in self.__topology:
            self.__nodes[id].receive(internal_result)
            internal_result[id] = self.__nodes[id].output()
        return internal_result


class LoopException(Exception):
    message = None

    def __init__(self, message):
        self.message = message

    def __str__(self):
        return repr(self.message)


class MissException(Exception):
    message = None

    def __init__(self, message):
        self.message = message

    def __str__(self):
        return repr(self.message)
