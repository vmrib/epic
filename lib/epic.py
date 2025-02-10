"""
Algorithms to approximate the neighborhood intersection of vertices in a graph.
Includes default, vertex-normalized and edge-normalized versions.
"""

from collections import defaultdict
import networkx as nx
from math import floor, ceil, log2, log
from itertools import combinations
from random import choice


# Constants used for sample size calculations
c = 0.5  # Used in ε-approximations
ć = 0.5  # Used in relative (p, ε)-approximations


def vertex_normalized_intersection(
    G: nx.Graph, ε: float, δ: float, allow_sample_overflow=False
) -> dict:
    """
    Estimates the vertex-normalized intersection of all pairs of vertices in a graph
    with absolute error ε and confidence 1 - δ.

    Parameters
    ----------
    G : nx.Graph
        A NetworkX graph.
    ε : float
        Accuracy paramenter.
    δ : float
        Confidence parameter.
    allow_sample_overflow : bool, optional
        Allow the number of samples to exceed the number of vertices in the graph, by default False

    Returns
    -------
    dict
        A dictionary of estimated vertex-normalized intersections.

    Raises
    ------
    ValueError
        If the maximum degree of the graph is zero or one.
    """
    î_vertex = defaultdict(float)
    node_list = list(G.nodes())
    Δ = get_maximum_degree(G)
    if Δ == 0 or Δ == 1:
        raise ValueError(
            "No intersection to estimate. Intersection between all pairs is zero."
        )

    m = ceil(c / ε**2 * (floor(2 * log2(Δ)) + log(1 / δ)))
    if (not allow_sample_overflow) and m > G.number_of_nodes():
        raise ValueError(
            f"The number of samples is greater than the number of vertices in the graph. {m} > {G.number_of_nodes()}"
        )

    for _ in range(m):
        s = sample_vertex(node_list)
        for n_i, n_j in combinations(G.neighbors(s), 2):
            î_vertex[(n_i, n_j) if n_i < n_j else (n_j, n_i)] += 1 / m

    return î_vertex


def edge_normalized_intersection(
    G: nx.Graph, ε: float, δ: float, allow_sample_overflow=False
) -> dict:
    """
    Estimates the edge-normalized intersection of all pairs of vertices in a graph
    with absolute error ε and confidence 1 - δ.

    Parameters
    ----------
    G : nx.Graph
        A NetworkX graph.
    ε : float
        Accuracy paramenter.
    δ : float
        Confidence parameter.
    allow_sample_overflow : bool, optional
        Allow the number of samples to exceed the number of edges in the graph, by default False

    Returns
    -------
    dict
        A dictionary of estimated edge-normalized intersections.

    Raises
    ------
    ValueError
        If the maximum degree of the graph is zero or one.
    """
    î_edge = defaultdict(float)
    edge_list = list(G.edges())
    Δ = get_maximum_degree(G)
    if Δ == 0 or Δ == 1:
        raise ValueError(
            "No intersection to estimate. Intersection between all pairs is zero."
        )

    m = ceil(c / ε**2 * (floor(log2(Δ)) + 2 + log(1 / δ)))
    if (not allow_sample_overflow) and m > G.number_of_edges():
        raise ValueError(
            f"The number of samples is greater than the number of edges in the graph. {m} > {G.number_of_edges()}"
        )

    for _ in range(m):
        (u, v) = sample_edge(edge_list)
        for u_i in G.neighbors(u):
            if u_i != v:
                î_edge[(u_i, v) if u_i < v else (v, u_i)] += 1 / m

        for v_i in G.neighbors(v):
            if v_i != u:
                î_edge[(v_i, u) if v_i < u else (u, v_i)] += 1 / m

    return î_edge


def intersection_vertex_sampling(
    G: nx.Graph, ε: float, δ: float, p: float, allow_sample_overflow=False
) -> dict:
    """
    Estimates the intersection of all pairs of vertices in a graph
    (by sampling vertices) with relative error ε and confidence 1 - δ.

    Parameters
    ----------
    G : nx.Graph
        A NetworkX graph.
    ε : float
        Accuracy paramenter.
    δ : float
        Confidence parameter.
    p : float
        Threshold parameter.
    allow_sample_overflow : bool, optional
        Allow the number of samples to exceed the number of vertices in the graph, by default False

    Returns
    -------
    dict
        A dictionary of estimated intersections.

    Raises
    ------
    ValueError
        If the maximum degree of the graph is zero or one, or no estimate is found.
    """
    î = defaultdict(float)
    nodes_list = list(G.nodes())
    Δ = get_maximum_degree(G)
    if Δ == 0 or Δ == 1:
        raise ValueError(
            "No intersection to estimate. Intersection between all pairs is zero."
        )

    m = ceil((ć / (ε**2 * p)) * (floor(2 * log2(Δ)) * log(1 / p) + log(1 / δ)))
    if (not allow_sample_overflow) and m > G.number_of_nodes():
        raise ValueError(
            f"The number of samples is greater than the number of vertices in the graph. {m} > {G.number_of_edges()}"
        )

    normalizer = G.number_of_nodes() / m
    for _ in range(m):
        s = sample_vertex(nodes_list)
        for n_i, n_j in combinations(G.neighbors(s), 2):
            î[(n_i, n_j) if n_i < n_j else (n_j, n_i)] += normalizer

    return î


def intersection_edge_sampling(
    G: nx.Graph, ε: float, δ: float, p: float, allow_sample_overflow=False
) -> dict:
    """
    Estimates the intersection of all pairs of vertices in a graph
    (by sampling edges) with relative error ε and confidence 1 - δ.

    Parameters
    ----------
    G : nx.Graph
        A NetworkX graph.
    ε : float
        Accuracy paramenter.
    δ : float
        Confidence parameter.
    p : float
        Threshold parameter.
    allow_sample_overflow : bool, optional
        Allow the number of samples to exceed the number of edges in the graph, by default False

    Returns
    -------
    dict
        A dictionary of estimated intersections.

    Raises
    ------
    ValueError
        If the maximum degree of the graph is zero or one, or no estimate is found.
    """
    î = defaultdict(float)
    edges_list = list(G.edges())

    Δ = get_maximum_degree(G)
    if Δ == 0 or Δ == 1:
        raise ValueError(
            "No intersection to estimate. Intersection between all pairs is zero."
        )

    m = ceil((ć / (ε**2 * p)) * ((floor(log2(Δ)) + 2) * log(1 / p) + log(1 / δ)))
    if (not allow_sample_overflow) and m > G.number_of_edges():
        raise ValueError(
            f"The number of samples is greater than the number of edges in the graph. {m} > {G.number_of_edges()}"
        )

    normalizer = G.number_of_edges() / (2 * m)
    for _ in range(m):
        (u, v) = sample_edge(edges_list)
        for u_i in G.neighbors(u):
            if u_i != v:
                î[(u_i, v) if u_i < v else (v, u_i)] += normalizer

        for v_i in G.neighbors(v):
            if v_i != u:
                î[(v_i, u) if v_i < u else (u, v_i)] += normalizer

    return î


def get_maximum_degree(G: nx.Graph) -> int:
    """
    Compute the maximum degree of a graph.

    Parameters
    ----------
    G : nx.Graph
        A NetworkX graph.

    Returns
    -------
    int
        The maximum degree of the graph.
    """
    return max(dict(G.degree()).values()) if G.number_of_nodes() else 0


def sample_vertex(nodes: list) -> int:
    """
    Sample a vertex uniformly from a list of nodes.

    Parameters
    ----------
    nodes : list
        A list of nodes.

    Returns
    -------
    int
        A vertex in the list of nodes.
    """
    return choice(nodes)


def sample_edge(edges: list) -> tuple:
    """
    Sample an edge uniformly from a graph.

    Parameters
    ----------
    edges : list
        A list of edges.

    Returns
    -------
    tuple
        An edge in the graph.
    """
    return choice(edges)
