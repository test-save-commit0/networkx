from itertools import combinations
import networkx as nx
__all__ = ['dispersion']


@nx._dispatchable
def dispersion(G, u=None, v=None, normalized=True, alpha=1.0, b=0.0, c=0.0):
    """Calculate dispersion between `u` and `v` in `G`.

    A link between two actors (`u` and `v`) has a high dispersion when their
    mutual ties (`s` and `t`) are not well connected with each other.

    Parameters
    ----------
    G : graph
        A NetworkX graph.
    u : node, optional
        The source for the dispersion score (e.g. ego node of the network).
    v : node, optional
        The target of the dispersion score if specified.
    normalized : bool
        If True (default) normalize by the embeddedness of the nodes (u and v).
    alpha, b, c : float
        Parameters for the normalization procedure. When `normalized` is True,
        the dispersion value is normalized by::

            result = ((dispersion + b) ** alpha) / (embeddedness + c)

        as long as the denominator is nonzero.

    Returns
    -------
    nodes : dictionary
        If u (v) is specified, returns a dictionary of nodes with dispersion
        score for all "target" ("source") nodes. If neither u nor v is
        specified, returns a dictionary of dictionaries for all nodes 'u' in the
        graph with a dispersion score for each node 'v'.

    Notes
    -----
    This implementation follows Lars Backstrom and Jon Kleinberg [1]_. Typical
    usage would be to run dispersion on the ego network $G_u$ if $u$ were
    specified.  Running :func:`dispersion` with neither $u$ nor $v$ specified
    can take some time to complete.

    References
    ----------
    .. [1] Romantic Partnerships and the Dispersion of Social Ties:
        A Network Analysis of Relationship Status on Facebook.
        Lars Backstrom, Jon Kleinberg.
        https://arxiv.org/pdf/1310.6753v1.pdf

    """
    pass
