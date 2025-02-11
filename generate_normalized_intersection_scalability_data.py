"""
Generate data for analyzing the scalability of normalized intersection algorithms (vertex and edge normalized). The data includes statistical measures (mean, standard deviation, max, min) of the runtime of the exact and approximate algorithms.

This data is used in Figure 5 of the paper.

Example usage:
python generate_normalized_intersection_scalability_data.py --epsilon 0.05 --delta 0.1 --n 10 --barabasi_m 10 --barabasi_ns 10000 20000 30000 40000 50000 60000 80000 90000 100000

This will generate a CSV file in the `./results` directory containing the data for the normalized intersection scalability analysis.
"""

import argparse
from pathlib import Path
from alive_progress import alive_bar
import networkx as nx
import pandas as pd
import time
import lib.intersection as intersection
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


def stopwatch(func):
    """
    A wrapper to measure the runtime of a function. The wrapped function returns the result and the runtime.

    Parameters
    ----------
    func : function
        The function to be measured.

    Returns
    -------
    function
        The wrapped function.
    """

    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()

        return result, end - start

    return wrapper


def generate_data(
    barabasi_ns: list[int], barabasi_m: int, ε: float, δ: float, n: int
) -> pd.DataFrame:
    """
    Generates data for analysing the scalability of normalized intersection algorithms (vertex and edge normalized).

    The data includes statistical measures (mean, standard deviation, max, min) of the runtime of the exact and approximate algorithms.

    Parameters
    ----------
    barabasi_ns : list[int]
        A list of number of vertices for the Barabasi-Albert graphs.
    barabasi_m : int
        The fixed number of edges to attach from a new node to an existing node in the Barabasi-Albert graphs.
    ε : float
        Accuracy parameter.
    δ : float
        Confidence parameter.
    n : int
        Number of iterations used to compute the statistics.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the results.
    """

    i_vertex_timed = stopwatch(intersection.vertex_normalized_intersection)
    î_vertex_timed = stopwatch(epic.vertex_normalized_intersection)
    i_edge_timed = stopwatch(intersection.edge_normalized_intersection)
    î_edge_timed = stopwatch(epic.edge_normalized_intersection)

    i_vertex_mean_runtime_values = []
    i_vertex_stddev_runtime_values = []
    i_vertex_max_runtime_values = []
    i_vertex_min_runtime_values = []
    î_vertex_mean_runtime_values = []
    î_vertex_stddev_runtime_values = []
    î_vertex_max_runtime_values = []
    î_vertex_min_runtime_values = []
    i_edge_mean_runtime_values = []
    i_edge_stddev_runtime_values = []
    i_edge_max_runtime_values = []
    i_edge_min_runtime_values = []
    î_edge_mean_runtime_values = []
    î_edge_stddev_runtime_values = []
    î_edge_max_runtime_values = []
    î_edge_min_runtime_values = []

    with alive_bar(len(barabasi_ns) * n) as bar:
        for barabasi_n in barabasi_ns:
            G = nx.barabasi_albert_graph(barabasi_n, barabasi_m)

            i_vertex_runtime_values = []
            î_vertex_runtime_values = []
            i_edge_runtime_values = []
            î_edge_runtime_values = []

            for _ in range(n):
                i_vertex, i_vertex_runtime = i_vertex_timed(G)
                i_vertex_runtime_values.append(i_vertex_runtime)

                î_vertex, î_vertex_runtime = î_vertex_timed(
                    G, ε, δ, allow_sample_overflow=True
                )
                î_vertex_runtime_values.append(î_vertex_runtime)

                i_edge, i_edge_runtime = i_edge_timed(G)
                i_edge_runtime_values.append(i_edge_runtime)

                î_edge, î_edge_runtime = î_edge_timed(
                    G, ε, δ, allow_sample_overflow=True
                )
                î_edge_runtime_values.append(î_edge_runtime)

                bar()

            s_i_vertex_runtime = pd.Series(i_vertex_runtime_values)
            i_vertex_mean_runtime_values.append(s_i_vertex_runtime.mean())
            i_vertex_stddev_runtime_values.append(s_i_vertex_runtime.std())
            i_vertex_max_runtime_values.append(s_i_vertex_runtime.max())
            i_vertex_min_runtime_values.append(s_i_vertex_runtime.min())

            s_î_vertex_runtime = pd.Series(î_vertex_runtime_values)
            î_vertex_mean_runtime_values.append(s_î_vertex_runtime.mean())
            î_vertex_stddev_runtime_values.append(s_î_vertex_runtime.std())
            î_vertex_max_runtime_values.append(s_î_vertex_runtime.max())
            î_vertex_min_runtime_values.append(s_î_vertex_runtime.min())

            s_i_edge_runtime = pd.Series(i_edge_runtime_values)
            i_edge_mean_runtime_values.append(s_i_edge_runtime.mean())
            i_edge_stddev_runtime_values.append(s_i_vertex_runtime.std())
            i_edge_max_runtime_values.append(s_i_vertex_runtime.max())
            i_edge_min_runtime_values.append(s_i_vertex_runtime.min())

            s_î_edge_runtime = pd.Series(î_edge_runtime_values)
            î_edge_mean_runtime_values.append(s_î_edge_runtime.mean())
            î_edge_stddev_runtime_values.append(s_î_vertex_runtime.std())
            î_edge_max_runtime_values.append(s_î_vertex_runtime.max())
            î_edge_min_runtime_values.append(s_î_vertex_runtime.min())

    return pd.DataFrame(
        {
            "Barabasi n": barabasi_ns,
            "Exact i_vertex Mean Runtime": i_vertex_mean_runtime_values,
            "Exact i_vertex StdDev Runtime": i_vertex_stddev_runtime_values,
            "Exact i_vertex Max Runtime": i_vertex_max_runtime_values,
            "Exact i_vertex Min Runtime": i_vertex_min_runtime_values,
            "î_vertex Mean Runtime": î_vertex_mean_runtime_values,
            "î_vertex StdDev Runtime": î_vertex_stddev_runtime_values,
            "î_vertex Max Runtime": î_vertex_max_runtime_values,
            "î_vertex Min Runtime": î_vertex_min_runtime_values,
            "Exact i_edge Mean Runtime": i_edge_mean_runtime_values,
            "Exact i_edge StdDev Runtime": i_edge_stddev_runtime_values,
            "Exact i_edge Max Runtime": i_edge_max_runtime_values,
            "Exact i_edge Min Runtime": i_edge_min_runtime_values,
            "î_edge Mean Runtime": î_edge_mean_runtime_values,
            "î_edge StdDev Runtime": î_edge_stddev_runtime_values,
            "î_edge Max Runtime": î_edge_max_runtime_values,
            "î_edge Min Runtime": î_edge_min_runtime_values,
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate data for analyzing the scalability of normalized intersection algorithms (vertex and edge normalized).",
        epilog="Example usage: python generate_normalized_intersection_scalability_data.py --epsilon 0.05 --delta 0.1 --n 10 --barabasi_m 10 --barabasi_ns 10000 20000 30000 40000 50000 60000 80000 90000 100000",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--epsilon", type=float, default=0.05, help="Accuracy parameter"
    )
    parser.add_argument("--delta", type=float, default=0.1, help="Confidence parameter")
    parser.add_argument("--n", type=int, default=10, help="Number of iterations")
    parser.add_argument(
        "--barabasi_m",
        type=int,
        default=10,
        help="Fixed number of edges to attach from a new node to an existing node in the Barabasi-Albert graphs",
    )
    parser.add_argument(
        "--barabasi_ns",
        type=int,
        nargs="+",
        default=[
            10_000,
            20_000,
            30_000,
            40_000,
            50_000,
            60_000,
            80_000,
            90_000,
            100_000,
        ],
        help="Number of nodes for each Barabasi-Albert graph",
    )

    args = parser.parse_args()

    Path("./results").mkdir(parents=True, exist_ok=True)
    output_path = f"./results/normalized_intersection_scalability_data_epsilon_{args.epsilon}_delta_{args.delta}_n_{args.n}_barabasi_m_{args.barabasi_m}.csv"

    df = generate_data(
        args.barabasi_ns, args.barabasi_m, args.epsilon, args.delta, args.n
    )
    df.to_csv(output_path, index=False)
