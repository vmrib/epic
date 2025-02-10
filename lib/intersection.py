"""
Functions to compute the (exact) neighborhood intersection of vertices in a graph.
Includes default, vertex-normalized and edge-normalized versions.
"""

from collections import defaultdict
from itertools import combinations
import networkx as nx
import numpy as np


def intersection(G: nx.Graph) -> dict:
    """
    Compute the neighborhood intersection of all pairs of vertices in a graph.

    Parameters
    ----------
    G : nx.Graph
        A NetworkX graph.

    Returns
    -------
    dict
        A dictionary of intersections.
    """
    pairs = defaultdict(int)
    for u in G.nodes():
        for n_i, n_j in combinations(G.neighbors(u), 2):
            pairs[(n_i, n_j) if n_i < n_j else (n_j, n_i)] += 1

    return pairs


def vertex_normalized_intersection(G: nx.Graph) -> dict:
    """
    Compute the vertex-normalized intersection of all pairs of vertices in a graph.

    Parameters
    ----------
    G : nx.Graph
        A NetworkX graph.

    Returns
    -------
    dict
        A dictionary of vertex-normalized intersections.
    """
    pairs = defaultdict(int)
    V_size = G.number_of_nodes()
    for u in G.nodes():
        for n_i, n_j in combinations(G.neighbors(u), 2):
            pairs[(n_i, n_j) if n_i < n_j else (n_j, n_i)] += 1 / V_size

    return pairs


def edge_normalized_intersection(G: nx.Graph) -> dict:
    """
    Compute the edge-normalized intersection of all pairs of vertices in a graph.

    Parameters
    ----------
    G : nx.Graph
        A NetworkX graph.

    Returns
    -------
    dict
        A dictionary of edge-normalized intersections.
    """
    pairs = defaultdict(int)
    E_size = G.number_of_edges()
    for u in G.nodes():
        for n_i, n_j in combinations(G.neighbors(u), 2):
            pairs[(n_i, n_j) if n_i < n_j else (n_j, n_i)] += 2 / E_size

    return pairs
