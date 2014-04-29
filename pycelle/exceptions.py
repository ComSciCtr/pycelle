# -*- coding: utf-8 -*-

"""
Exceptions

"""

__all__ = [
    'PycelleException',
]

class PycelleException(Exception):
    """
    Base class for all `pycelle` exceptions.

    """
    def __init__(self, *args, **kwargs):
        if 'msg' in kwargs:
            # Override the message in the first argument.
            self.msg = kwargs['msg']
        elif args:
            self.msg = args[0]
        else:
            self.msg = ''
        self.args = args
        self.kwargs = kwargs

    def __str__(self):
        return self.msg

    def __repr__(self):
        return "{0}{1}".format(self.__class__.__name__, repr(self.args))
