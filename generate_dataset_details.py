"""
Generate data containg the number of nodes, number of edges, maximum degree, and non-empty intersections ratio for multiple graphs.

This data is used in Table 3 of the paper.

Example usage:
python3 generate_dataset_details.py graphs/as20000102.txt graphs/facebook_new_sites_edges.csv graphs/facebook_new_sites_edges.csv

This will generate a CSV file in the `./results` directory containing the data for the graphs as20000102.txt, facebook_new_sites_edges.csv, and facebook_new_sites_edges.csv.
"""

import argparse
import math
from pathlib import Path
from alive_progress import alive_bar
import networkx as nx
import pandas as pd
from lib.epic import get_maximum_degree
from lib.intersection import intersection


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


def calculate_non_empty_intersections_ratio(G: nx.Graph) -> float:
    """
    Calculates the ratio of non-empty neighborhood intersections to all possible neighborhood intersections (pairs of vertices).

    Parameters
    ----------
    G : nx.Graph
        A NetworkX graph.

    Returns
    -------
    float
        The ratio of non-empty neighborhood intersections to all possible neighborhood intersections.
    """
    number_of_pairs = math.comb(G.number_of_nodes(), 2)
    non_empty_intersections = len(intersection(G))

    return non_empty_intersections / number_of_pairs


def generate_data(file_paths: list[Path]) -> pd.DataFrame:
    """
    Generates data containg the number of nodes, number of edges, maximum degree, and non-empty intersections ratio for each graph.

    Each path is assumed to be a graph file, in the format of an edgelist. The file can be either a CSV or a text file.

    Parameters
    ----------
    file_paths : list[Path]
        List of paths to the graph files.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the data.
    """
    graph_name_values = []
    number_of_nodes_values = []
    number_of_edges_values = []
    maximum_degree_values = []
    non_empty_intersections_ratio_values = []

    with alive_bar(len(file_paths)) as bar:
        for file_path in file_paths:
            G = read_graph(
                file_path, delimiter="," if file_path.endswith(".csv") else None
            )
            graph_name_values.append(Path(file_path).stem)
            number_of_nodes_values.append(G.number_of_nodes())
            number_of_edges_values.append(G.number_of_edges())
            maximum_degree_values.append(get_maximum_degree(G))
            non_empty_intersections_ratio_values.append(
                calculate_non_empty_intersections_ratio(G)
            )
            bar()

    return pd.DataFrame(
        {
            "Graph": graph_name_values,
            "Number of Nodes": number_of_nodes_values,
            "Number of Edges": number_of_edges_values,
            "Maximum Degree": maximum_degree_values,
            "Non-Empty Intersections Ratio": non_empty_intersections_ratio_values,
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generates data containg the number of nodes, number of edges, maximum degree, and non-empty intersections ratio for multiple graphs.",
        epilog="Example usage: python generate_dataset_details.py graphs/as20000102.txt graphs/facebook_new_sites_edges.csv",
    )
    parser.add_argument(
        "graph_paths",
        type=str,
        nargs="+",
        help="List of paths. Each path is assumed to be a graph file, in the format of an edgelist. The file can be either a CSV or a text file.",
    )
    args = parser.parse_args()

    Path("./results").mkdir(parents=True, exist_ok=True)
    output_path = "./results/dataset_details.csv"

    data = generate_data(args.graph_paths)
    data.to_csv(output_path, index=False)
