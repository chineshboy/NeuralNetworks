# NeuralNetworks
a simple version of neural networks

Some fixed settings:

Each neural node must have its id, which is a string. You can assign any string as an id.
The input node in this system is virtual. That means they're just used to identify the inputs. Only the internal and output nodes can be(and must be) trained.
The activation function is fixed as max(0, x).

What you can do:

You can change the default weight for every input of every node.
You can change teh learning rate to any float number.
You can add any number of input nodes, internal nodes, output nodes.
You can specify the relation of the nodes.
You can specify initial different weights or theta values for each node separately.
You can train with any number of test sets.
You can compute the prediction with trained model.
You can store the trained model into file and reload or modify it later.

Usage:

First of all, initialize a Configuration.
Use add_input_node, add_output_node to specify the input/output node ids.
Use add_relation to build the structure of the network.
Use set_default_weight, set_learn_rate to set these attributes. If unset, default weight is 0.5 and default learning rate is 0.01.
Use add_weight, add_theta to set different values to each node.

After all the configuration set, initialize a Network with this Configuration.
Use train to train the network. In this function, you should specify what value is got from each input node by a dict, and what value is expected to be given by each output node by another dict.
Use predict to get a result from a trained model. The input and output are similar to the two input of the train function. Besides the output of each output node, you can also get the output of each internal node.
Use dump to get a Configuration of current status.
Use dump_to_file to store a Configuration to file, and then you can use constructor to reload again.

Example:

A simple example can be found in test.py, which is used to train a model with logical add.
Another example shows how to do model dump and reload. You can modify the reloaded model as you like.
