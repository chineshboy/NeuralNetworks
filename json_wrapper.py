import json

__author__ = 'jianxun'


def encode(target, encoding='utf-8'):
    """encode a unicode string or all unicode strings inside a list/dict recursively

    :param target: an object, could be a dict/list/string
    :param encoding: the encodings supported python
    :return:
    """
    if isinstance(target, dict):
        return {encode(key, encoding): encode(value, encoding) for key, value in target.iteritems()}
    elif isinstance(target, list):
        return [encode(element, encoding) for element in target]
    elif isinstance(target, unicode):
        return target.encode(encoding)
    else:
        return target


def loads(json_string, encoding='utf-8'):
    """json.loads returns `unicode object` for a string, you can use this function to do encoding

    :param json_string: a well-formed string
    :param encoding: a string that represents the encoding you want, like 'utf-8', 'gbk'...
    :return: an object that the json_string represents
    """
    result = json.loads(json_string)
    return encode(result, encoding)


def dumps(object):
    """ dumps an object to a json string

    :param object: the object to be dumped
    :return: a json string represents the object
    """
    return json.dumps(object)
