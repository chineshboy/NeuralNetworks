from SimpleNN import Configuration, Network


def test_simple_nnp():
    conf = Configuration()
    conf.add_input_node(['x', 'y'])
    conf.add_relation('x', 'l11')
    conf.add_relation('x', 'l12')
    conf.add_relation('x', 'l13')
    conf.add_relation('y', 'l11')
    conf.add_relation('y', 'l12')
    conf.add_relation('y', 'l13')
    conf.add_relation('l11', 'o')
    conf.add_relation('l12', 'o')
    conf.add_relation('l13', 'o')
    conf.add_output_node('o')
    # conf.set_learn_rate(0.1)
    net = Network(conf)
    # train 'and' operation
    for i in xrange(10000):
        net.train({'x': 1, 'y': 1}, {'o': 1})
        net.train({'x': 1, 'y': 0}, {'o': 0})
        net.train({'x': 0, 'y': 1}, {'o': 0})
        net.train({'x': 0, 'y': 0}, {'o': 0})

    print net.predict({'x': 1, 'y': 1})
    print net.predict({'x': 1, 'y': 0})
    print net.predict({'x': 0, 'y': 1})
    print net.predict({'x': 0, 'y': 0})


def test_dump():
    conf = Configuration()
    conf.add_input_node(['x', 'y'])
    conf.add_relation('x', 'l11')
    conf.add_relation('x', 'l12')
    conf.add_relation('x', 'l13')
    conf.add_relation('y', 'l11')
    conf.add_relation('y', 'l12')
    conf.add_relation('y', 'l13')
    conf.add_relation('l11', 'o')
    conf.add_relation('l12', 'o')
    conf.add_relation('l13', 'o')
    conf.add_output_node('o')
    net = Network(conf)
    for i in xrange(10000):
        net.train({'x': 1, 'y': 1}, {'o': 1})
        net.train({'x': 1, 'y': 0}, {'o': 1})
        net.train({'x': 0, 'y': 1}, {'o': 1})
        net.train({'x': 0, 'y': 0}, {'o': 0})

    print net.predict({'x': 1, 'y': 1})
    print net.predict({'x': 1, 'y': 0})
    print net.predict({'x': 0, 'y': 1})
    print net.predict({'x': 0, 'y': 0})
    temp_conf = net.dump()
    temp_conf.dump_to_file('./dump_conf')
    new_conf = Configuration('./dump_conf')
    new_net = Network(new_conf)
    print new_net.predict({'x': 1, 'y': 1})
    print new_net.predict({'x': 1, 'y': 0})
    print new_net.predict({'x': 0, 'y': 1})
    print new_net.predict({'x': 0, 'y': 0})


if __name__ == '__main__':
    test_simple_nnp()
    test_dump()
