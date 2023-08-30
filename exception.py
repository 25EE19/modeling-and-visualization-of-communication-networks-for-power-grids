""" Defines the Exceptions needed """

class ParameterException(Exception):
    """ Customizable Exception for wrong Parameters in params.py """
    pass


class ValueException(Exception):
    """ Cusomizable Exception for value Errors, like devision by 0 """
    pass


class GridException(Exception):
    """ Will be raised if there is an error during a grid test """
    pass
