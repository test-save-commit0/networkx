"""Functions which help end users define customize node_match and
edge_match functions to use during isomorphism checks.
"""
import math
import types
from itertools import permutations
__all__ = ['categorical_node_match', 'categorical_edge_match',
    'categorical_multiedge_match', 'numerical_node_match',
    'numerical_edge_match', 'numerical_multiedge_match',
    'generic_node_match', 'generic_edge_match', 'generic_multiedge_match']


def copyfunc(f, name=None):
    """Returns a deepcopy of a function."""
    return types.FunctionType(f.__code__, f.__globals__, name or f.__name__,
                              f.__defaults__, f.__closure__)


def allclose(x, y, rtol=1e-05, atol=1e-08):
    """Returns True if x and y are sufficiently close, elementwise.

    Parameters
    ----------
    rtol : float
        The relative error tolerance.
    atol : float
        The absolute error tolerance.

    """
    return all(math.isclose(a, b, rel_tol=rtol, abs_tol=atol) for a, b in zip(x, y))


categorical_doc = """
Returns a comparison function for a categorical node attribute.

The value(s) of the attr(s) must be hashable and comparable via the ==
operator since they are placed into a set([]) object.  If the sets from
G1 and G2 are the same, then the constructed function returns True.

Parameters
----------
attr : string | list
    The categorical node attribute to compare, or a list of categorical
    node attributes to compare.
default : value | list
    The default value for the categorical node attribute, or a list of
    default values for the categorical node attributes.

Returns
-------
match : function
    The customized, categorical `node_match` function.

Examples
--------
>>> import networkx.algorithms.isomorphism as iso
>>> nm = iso.categorical_node_match("size", 1)
>>> nm = iso.categorical_node_match(["color", "size"], ["red", 2])

"""
categorical_edge_match = copyfunc(categorical_node_match,
    'categorical_edge_match')
categorical_node_match.__doc__ = categorical_doc
categorical_edge_match.__doc__ = categorical_doc.replace('node', 'edge')
tmpdoc = categorical_doc.replace('node', 'edge')
tmpdoc = tmpdoc.replace('categorical_edge_match', 'categorical_multiedge_match'
    )
categorical_multiedge_match.__doc__ = tmpdoc
numerical_doc = """
Returns a comparison function for a numerical node attribute.

The value(s) of the attr(s) must be numerical and sortable.  If the
sorted list of values from G1 and G2 are the same within some
tolerance, then the constructed function returns True.

Parameters
----------
attr : string | list
    The numerical node attribute to compare, or a list of numerical
    node attributes to compare.
default : value | list
    The default value for the numerical node attribute, or a list of
    default values for the numerical node attributes.
rtol : float
    The relative error tolerance.
atol : float
    The absolute error tolerance.

Returns
-------
match : function
    The customized, numerical `node_match` function.

Examples
--------
>>> import networkx.algorithms.isomorphism as iso
>>> nm = iso.numerical_node_match("weight", 1.0)
>>> nm = iso.numerical_node_match(["weight", "linewidth"], [0.25, 0.5])

"""
numerical_edge_match = copyfunc(numerical_node_match, 'numerical_edge_match')
numerical_node_match.__doc__ = numerical_doc
numerical_edge_match.__doc__ = numerical_doc.replace('node', 'edge')
tmpdoc = numerical_doc.replace('node', 'edge')
tmpdoc = tmpdoc.replace('numerical_edge_match', 'numerical_multiedge_match')
numerical_multiedge_match.__doc__ = tmpdoc
generic_doc = """
Returns a comparison function for a generic attribute.

The value(s) of the attr(s) are compared using the specified
operators. If all the attributes are equal, then the constructed
function returns True.

Parameters
----------
attr : string | list
    The node attribute to compare, or a list of node attributes
    to compare.
default : value | list
    The default value for the node attribute, or a list of
    default values for the node attributes.
op : callable | list
    The operator to use when comparing attribute values, or a list
    of operators to use when comparing values for each attribute.

Returns
-------
match : function
    The customized, generic `node_match` function.

Examples
--------
>>> from operator import eq
>>> from math import isclose
>>> from networkx.algorithms.isomorphism import generic_node_match
>>> nm = generic_node_match("weight", 1.0, isclose)
>>> nm = generic_node_match("color", "red", eq)
>>> nm = generic_node_match(["weight", "color"], [1.0, "red"], [isclose, eq])

"""
generic_edge_match = copyfunc(generic_node_match, 'generic_edge_match')


def generic_multiedge_match(attr, default, op):
    """Returns a comparison function for a generic attribute.

    The value(s) of the attr(s) are compared using the specified
    operators. If all the attributes are equal, then the constructed
    function returns True. Potentially, the constructed edge_match
    function can be slow since it must verify that no isomorphism
    exists between the multiedges before it returns False.

    Parameters
    ----------
    attr : string | list
        The edge attribute to compare, or a list of node attributes
        to compare.
    default : value | list
        The default value for the edge attribute, or a list of
        default values for the edgeattributes.
    op : callable | list
        The operator to use when comparing attribute values, or a list
        of operators to use when comparing values for each attribute.

    Returns
    -------
    match : function
        The customized, generic `edge_match` function.

    Examples
    --------
    >>> from operator import eq
    >>> from math import isclose
    >>> from networkx.algorithms.isomorphism import generic_node_match
    >>> nm = generic_node_match("weight", 1.0, isclose)
    >>> nm = generic_node_match("color", "red", eq)
    >>> nm = generic_node_match(["weight", "color"], [1.0, "red"], [isclose, eq])

    """
    if isinstance(attr, str):
        attr = [attr]
        default = [default]
        op = [op]
    elif len(attr) != len(default) or len(attr) != len(op):
        raise ValueError("attr, default, and op must have the same length")

    def match(d1, d2):
        for a, def_val, operator in zip(attr, default, op):
            v1 = d1.get(a, def_val)
            v2 = d2.get(a, def_val)

            if not operator(v1, v2):
                return False

        return True

    def edge_match(e1, e2):
        if len(e1) != len(e2):
            return False

        for attrs1, attrs2 in permutations(e1.values(), len(e2)):
            if all(match(attrs1, attrs2) for attrs1, attrs2 in zip(e1.values(), e2.values())):
                return True

        return False

    return edge_match


generic_node_match.__doc__ = generic_doc
generic_edge_match.__doc__ = generic_doc.replace('node', 'edge')
