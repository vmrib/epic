"""
Generate data for heatmaps visualizing the Minimum Threshold p for vertex and edge sampling strategies on GNP graphs. The heatmaps display the Minimum Threshold p for varying graph sizes (number of vertices) and edge probabilities. Each cell in the heatmap represents the average Minimum Threshold p averaged over n iterations.

This data is used in Figures 6 and 7 of the paper.

Example usage:
python3 generate_heatmaps_data.py --epsilon 0.1 --delta 0.1 --n 10 --gnp_ns 1000 2000 3000 4000 5000 --gnp_ps 0.05 0.1 0.3 0.5 0.7 0.9

This will generate two CSV files in the `./results` directory containing the data for the vertex and edge sampling strategies heatmaps.
"""

import argparse
from pathlib import Path
from alive_progress import alive_bar
import networkx as nx
import pandas as pd
from math import floor, log2
from scipy.special import lambertw
from statistics import mean
import lib.epic as epic


def read_graph(file_path: any, delimiter: str | None = None) -> nx.Graph:
    """
    Reads a graph from an edgelist file and converts the node labels to integers.

    Parameters
    ----------
    file_path : any
        The path to the file.
    delimiter : str, optional
        The delimiter used in the file, by default None

    Returns
    -------
    nx.Graph
        A NetworkX graph.
    """
    G_raw = nx.read_edgelist(file_path, delimiter=delimiter, nodetype=int)
    G = nx.convert_node_labels_to_integers(G_raw)
    return G


def calculate_minimum_threshold_p_vertex_sampling(
    number_of_nodes: int, ε: float, δ: float, Δ: int
) -> float:
    """
    Calculate the Minimum Threshold p for neighborhood intersection estimation (vertex sampling).

    Parameters
    ----------
    number_of_nodes : int
        Number of nodes in the graph.
    ε : float
        Accuracy parameter.
    δ : float
        Confidence parameter.
    Δ : int
        Maximum degree of the graph.

    Returns
    -------
    float
        The minimum threshold p.
    """
    m = number_of_nodes
    D = floor(2 * log2(Δ))
    T = 2 * m * (1 / δ) ** (1 / D) * ε**2
    p = D * lambertw(T / D) / (2 * ε**2 * m)
    return p.real


def calculate_minimum_threshold_p_edge_sampling(
    number_of_edges: int, ε: float, δ: float, Δ: int
) -> float:
    """
    Calculate the Minimum Threshold p for neighborhood intersection estimation (edge sampling).

    Parameters
    ----------
    number_of_edges : int
        Number of edges in the graph.
    ε : float
        Accuracy parameter.
    δ : float
        Confidence parameter.
    Δ : int
        Maximum degree of the graph.

    Returns
    -------
    float
        The minimum threshold p.
    """
    m = number_of_edges
    D = floor(log2(Δ)) + 2
    T = 2 * m * (1 / δ) ** (1 / D) * ε**2
    p = D * lambertw(T / D) / (2 * ε**2 * m)
    return p.real


import pandas as pd


def generate_data(
    gnp_ns: list[int], gnp_ps: list[float], ε: float, δ: float, n: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates data for heatmaps visualizing the Minimum Threshold p for vertex and edge sampling strategies on GNP graphs.

    The heatmaps display the Minimum Threshold p for varying graph sizes (number of vertices) and edge probabilities.  Each cell in the heatmap represents the average Minimum Threshold p averaged over n iterations.

    Parameters
    ----------
    gnp_ns : list[int]
        A list of number of vertices for the GNP graphs. These values will form the x-axis of the heatmaps.
    gnp_ps : list[float]
        A list of edge generation probabilities for the GNP graphs. These values will form the y-axis of the heatmaps.
    epsilon : float
        Accuracy parameter.
    delta : float
        Confidence parameter.
    n : int
        Number of iterations used to compute the average minimum threshold p for each combination of `GNP n` and `GNP p`.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        A tuple containing two pandas DataFrames:
        - The first DataFrame stores the data for the vertex sampling strategy heatmap.
        - The second DataFrame stores the data for the edge sampling strategy heatmap.

        Rows correspond to `gnp_ps` values and columns correspond to `gnp_ns` values.  Each cell contains the average minimum threshold p for the corresponding (GNP n, GNP p) pair.
    """
    heat_map_vertex_sampling = {}
    heat_map_edge_sampling = {}

    with alive_bar(len(gnp_ns) * len(gnp_ps) * n) as bar:
        for gnp_n in gnp_ns:
            heat_map_vertex_sampling[gnp_n] = []
            heat_map_edge_sampling[gnp_n] = []

            for gnp_p in gnp_ps:
                heat_map_vertex_sampling_values = []
                heat_map_edge_sampling_values = []

                for _ in range(n):
                    G = nx.gnp_random_graph(gnp_n, gnp_p)
                    heat_map_vertex_sampling_values.append(
                        calculate_minimum_threshold_p_vertex_sampling(
                            gnp_n, ε, δ, epic.get_maximum_degree(G)
                        )
                    )
                    heat_map_edge_sampling_values.append(
                        calculate_minimum_threshold_p_edge_sampling(
                            G.number_of_edges(), ε, δ, epic.get_maximum_degree(G)
                        )
                    )

                    bar()

                heat_map_vertex_sampling[gnp_n].append(
                    mean(heat_map_vertex_sampling_values)
                )
                heat_map_edge_sampling[gnp_n].append(
                    mean(heat_map_edge_sampling_values)
                )

    df_vertex_sampling = pd.DataFrame(heat_map_vertex_sampling, index=gnp_ps)
    df_edge_sampling = pd.DataFrame(heat_map_edge_sampling, index=gnp_ps)

    return df_vertex_sampling, df_edge_sampling


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate data for heatmaps visualizing the Minimum Threshold p for vertex and edge sampling strategies on GNP graphs. The heatmaps display the Minimum Threshold p for varying graph sizes (number of vertices) and edge probabilities. Each cell in the heatmap represents the average Minimum Threshold p averaged over n iterations.",
        epilog="Example usage: python generate_heatmaps_data.py --epsilon 0.1 --delta 0.1 --n 10 --gnp_ns 1000 2000 3000 4000 5000 --gnp_ps 0.05 0.1 0.3 0.5 0.7 0.9",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epsilon", type=float, default=0.1, help="Accuracy parameter")
    parser.add_argument("--delta", type=float, default=0.1, help="Confidence parameter")
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Number of iterations used to compute the average minimum threshold p for each combination of `GNP n` and `GNP p`",
    )
    parser.add_argument(
        "--gnp_ns",
        type=int,
        nargs="+",
        default=[1000, 2000, 3000, 4000, 5000],
        help="Number of vertices for GNP graphs. This values will form the x-axis of the heatmaps.",
    )
    parser.add_argument(
        "--gnp_ps",
        type=float,
        nargs="+",
        default=[0.05, 0.1, 0.3, 0.5, 0.7, 0.9],
        help="Edge generation probabilities for GNP graphs. This values will form the y-axis of the heatmaps.",
    )

    args = parser.parse_args()

    Path("./results").mkdir(parents=True, exist_ok=True)
    output_vertex_sampling = f"./results/heat_map_vertex_sampling_epsilon_{args.epsilon}_delta_{args.delta}_n_{args.n}_gnp_ps_{str(args.gnp_ps).replace('[', '').replace(']', '').replace(', ', '_')}_gnp_ns_{str(args.gnp_ns).replace('[', '').replace(']', '').replace(', ', '_')}.csv"

    output_edge_sampling = f"./results/heat_map_edge_sampling_epsilon_{args.epsilon}_delta_{args.delta}_n_{args.n}_gnp_ps_{str(args.gnp_ps).replace('[', '').replace(']', '').replace(', ', '_')}_gnp_ns_{str(args.gnp_ns).replace('[', '').replace(']', '').replace(', ', '_')}.csv"

    df_vertex_sampling, df_edge_sampling = generate_data(
        args.gnp_ns, args.gnp_ps, args.epsilon, args.delta, args.n
    )

    df_vertex_sampling.to_csv(output_vertex_sampling)
    df_edge_sampling.to_csv(output_edge_sampling)
