"""
Generate data for comparing vertex and edge sampling strategies on GNP graphs. The data contains statistical measures (mean, standard deviation, max, min) for the runtime and rate of successfull approximations for the vertex and edge strategies. The data is collected for various GNP graphs with different numbers of vertices (GNP n) and edge probabilities (GNP p). Two tables are generated: Table 1 varies the edge generation probability (GNP p) while keeping the number of vertices (GNP n) constant, and Table 2 varies the number of vertices (GNP n) while keeping the edge probability (GNP p) constant.

This data is used in Table 1 and 2 of the paper.

Example usage:
python generate_intersection_data.py --epsilon 0.1 --delta 0.1 --n 10 --p_vertex 0.7 --p_edge 0.04 --table1_gnp_n 1000 --table1_gnp_ps 0.1 0.3 0.5 0.7 0.9 --table2_gnp_ns 500 1000 2000 4000 --table2_gnp_p 0.5

This will generate two CSV files in the `./results` directory containing the data for the two tables.
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


def generate_row_data(
    G: nx.Graph,
    ε: float,
    δ: float,
    n: int,
    p_vertex: float,
    p_edge: float,
) -> dict:
    """
    Generates data for a single row in the comparison tables, containing runtime and success rate statistics for vertex and edge sampling.

    Parameters
    ----------
    G : nx.Graph
        The input graph.
    ε : float
        Accuracy parameter.
    δ : float
        Confidence parameter.
    n : int
        Number of iterations used to compute the statistics.
    p_vertex : float
        Threshold parameter for vertex sampling strategy.
    p_edge : float
        Threshold parameter for edge sampling strategy.

    Returns
    -------
    dict
        A dictionary containing the calculated statistics (mean, standard deviation, min, max) for runtime and success rate for both vertex and edge sampling.
    """

    intersection_timed = stopwatch(intersection.intersection)
    î_vertex_sampling_timed = stopwatch(epic.intersection_vertex_sampling)
    î_edge_sampling_timed = stopwatch(epic.intersection_edge_sampling)

    row = {}

    i, i_runtime = intersection_timed(G)
    row["Intersection Runtime"] = i_runtime
    print(f"Computed intersection in {i_runtime:.2f} seconds")

    î_vertex_sampling_runtime_values = []
    î_vertex_sampling_rate_values = []
    î_edge_sampling_runtime_values = []
    î_edge_sampling_rate_values = []

    for _ in range(n):
        î_vertex_sampling, î_vertex_sampling_runtime = î_vertex_sampling_timed(
            G, ε, δ, p_vertex, allow_sample_overflow=True
        )
        î_vertex_sampling_runtime_values.append(î_vertex_sampling_runtime)
        print(
            f"Computed intersection (vertex sampling) in {î_vertex_sampling_runtime:.2f} seconds"
        )

        î_vertex_sampling_rate = 0
        for u, v in i:
            if abs(î_vertex_sampling[u, v] - i[u, v]) <= ε * i[u, v]:
                î_vertex_sampling_rate += 1
        î_vertex_sampling_rate /= len(i)
        î_vertex_sampling_rate_values.append(î_vertex_sampling_rate)

        î_edge_sampling, î_edge_sampling_runtime = î_edge_sampling_timed(
            G, ε, δ, p_edge, allow_sample_overflow=True
        )
        î_edge_sampling_runtime_values.append(î_edge_sampling_runtime)
        print(
            f"Computed intersection (edge sampling) in {î_edge_sampling_runtime:.2f} seconds"
        )

        î_edge_sampling_rate = 0
        for u, v in i:
            if abs(î_edge_sampling[u, v] - i[u, v]) <= ε * i[u, v]:
                î_edge_sampling_rate += 1
        î_edge_sampling_rate /= len(i)
        î_edge_sampling_rate_values.append(î_edge_sampling_rate)

    s_î_vertex_sampling_runtime = pd.Series(î_vertex_sampling_runtime_values)
    row["Vertex Sampling Mean Runtime"] = s_î_vertex_sampling_runtime.mean()
    row["Vertex Sampling Stddev Runtime"] = s_î_vertex_sampling_runtime.std()
    row["Vertex Sampling Max Runtime"] = s_î_vertex_sampling_runtime.max()
    row["Vertex Sampling Min Runtime"] = s_î_vertex_sampling_runtime.min()

    s_î_vertex_sampling_rate = pd.Series(î_vertex_sampling_rate_values)
    row["Vertex Sampling Mean Rate"] = s_î_vertex_sampling_rate.mean()
    row["Vertex Sampling Stddev Rate"] = s_î_vertex_sampling_rate.std()
    row["Vertex Sampling Max Rate"] = s_î_vertex_sampling_rate.max()
    row["Vertex Sampling Min Rate"] = s_î_vertex_sampling_rate.min()

    s_î_edge_sampling_runtime = pd.Series(î_edge_sampling_runtime_values)
    row["Edge Sampling Mean Runtime"] = s_î_edge_sampling_runtime.mean()
    row["Edge Sampling Stddev Runtime"] = s_î_edge_sampling_runtime.std()
    row["Edge Sampling Max Runtime"] = s_î_edge_sampling_runtime.max()
    row["Edge Sampling Min Runtime"] = s_î_edge_sampling_runtime.min()

    s_î_edge_sampling_rate = pd.Series(î_edge_sampling_rate_values)
    row["Edge Sampling Mean Rate"] = s_î_edge_sampling_rate.mean()
    row["Edge Sampling Stddev Rate"] = s_î_edge_sampling_rate.std()
    row["Edge Sampling Max Rate"] = s_î_edge_sampling_rate.max()
    row["Edge Sampling Min Rate"] = s_î_edge_sampling_rate.min()

    return row


def generate_data(
    table1_gnp_n: int,
    table1_gnp_ps: list[float],
    table2_gnp_ns: list[int],
    table2_gnp_p: int,
    ε: float,
    δ: float,
    n: int,
    p_vertex: float,
    p_edge: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates data for comparing vertex and edge sampling strategies on GNP graphs.

    The data contains statistical measures (mean, standard deviation, max, min) for the runtime and rate of successfull approximations for the vertex and edge strategies. The data is collected for various GNP graphs with different numbers of vertices (GNP n) and edge probabilities (GNP p). Two tables are generated:

    * **Table 1:** Varies the edge generation probability (GNP p) while keeping the number of vertices (GNP n) constant.
    * **Table 2:** Varies the number of vertices (GNP n) while keeping the edge probability (GNP p) constant.

    Parameters
    ----------
    table1_gnp_n : int
        The fixed number of vertices (GNP n) for the GNP graphs used in Table 1.
    table1_gnp_ps : list[float]
        A list of edge generation probabilities (GNP p) for the GNP graphs used in Table 1.
    table2_gnp_ns : list[int]
        A list of number of vertices (GNP n) for the GNP graphs used in Table 2.
    table2_gnp_p : int
        The fixed edge generation probability (GNP p) for the GNP graphs used in Table 2.
    ε : float
        Accuracy parameter.
    δ : float
        Confidence parameter.
    n : int
        Number of iterations used to compute the statistics.
    p_vertex : float
        Threshold parameter for vertex sampling strategy.
    p_edge : float
        Threshold parameter for edge sampling strategy.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        A tuple containing two pandas DataFrames:
        - The first DataFrame contains the data for Table 1 (varying GNP p).
        - The second DataFrame contains the data for Table 2 (varying GNP n).

        Each DataFrame includes columns for the varied parameter, and the statistical measures (mean, standard deviation, min, max) for runtime and success rate for both vertex and edge sampling.
    """

    table1_rows = []
    table2_rows = []

    with alive_bar(len(table1_gnp_ps) + len(table2_gnp_ns)) as bar:
        for gnp_p in table1_gnp_ps:
            G = nx.gnp_random_graph(table1_gnp_n, gnp_p)
            print(f"Generated GNP_{table1_gnp_n}_{gnp_p}")
            table1_rows.append(
                {"p": gnp_p, **generate_row_data(G, ε, δ, n, p_vertex, p_edge)}
            )
            bar()

        for gnp_n in table2_gnp_ns:
            G = nx.gnp_random_graph(gnp_n, table2_gnp_p)
            print(f"Generated GNP_{gnp_n}_{table2_gnp_p}")
            table2_rows.append(
                {"n": gnp_n, **generate_row_data(G, ε, δ, n, p_vertex, p_edge)}
            )
            bar()

    df1 = pd.DataFrame.from_records(table1_rows)
    df2 = pd.DataFrame.from_records(table2_rows)

    return df1, df2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate data for comparing vertex and edge sampling strategies on GNP graphs. The data contains statistical measures (mean, standard deviation, max, min) for the runtime and rate of successfull approximations for the vertex and edge strategies. The data is collected for various GNP graphs with different numbers of vertices (GNP n) and edge probabilities (GNP p). Two tables are generated: Table 1 varies the edge generation probability (GNP p) while keeping the number of vertices (GNP n) constant, and Table 2 varies the number of vertices (GNP n) while keeping the edge probability (GNP p) constant.",
        epilog="Example usage: python generate_intersection_data.py --epsilon 0.1 --delta 0.1 --n 10 --p_vertex 0.7 --p_edge 0.04 --table1_gnp_n 1000 --table1_gnp_ps 0.1 0.3 0.5 0.7 0.9 --table2_gnp_ns 500 1000 2000 4000 --table2_gnp_p 0.5",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epsilon", type=float, default=0.1, help="Accuracy parameter")
    parser.add_argument("--delta", type=float, default=0.1, help="Confidence parameter")
    parser.add_argument(
        "--p_vertex",
        type=float,
        default=0.7,
        help="Threshold parameter for vertex sampling",
    )
    parser.add_argument(
        "--p_edge",
        type=float,
        default=0.04,
        help="Threshold parameter for edge sampling",
    )
    parser.add_argument("--n", type=int, default=10, help="Number of iterations")
    parser.add_argument(
        "--table1_gnp_n",
        type=int,
        default=1000,
        help="Fixed number of vertices for Table 1",
    )
    parser.add_argument(
        "--table1_gnp_ps",
        type=float,
        nargs="+",
        default=[0.1, 0.3, 0.5, 0.7, 0.9],
        help="List of GNP edge probabilities for Table 1",
    )
    parser.add_argument(
        "--table2_gnp_ns",
        type=int,
        nargs="+",
        default=[500, 1000, 2000, 4000],
        help="List of GNP vertices for Table 2",
    )
    parser.add_argument(
        "--table2_gnp_p",
        type=float,
        default=0.5,
        help="Fixed GNP edge generation probability for Table 2",
    )

    args = parser.parse_args()
    Path("./results").mkdir(parents=True, exist_ok=True)

    table1_output_path = f"./results/intersection_data_table_1_gnp_n_{args.table1_gnp_n}_epsilon_{args.epsilon}_delta_{args.delta}_n_{args.n}_p_vertex_{args.p_vertex}_p_edge_{args.p_edge}.csv"
    table2_output_path = f"./results/intersection_data_table_2_gnp_p_{args.table2_gnp_p}_epsilon_{args.epsilon}_delta_{args.delta}_n_{args.n}_p_vertex_{args.p_vertex}_p_edge_{args.p_edge}.csv"

    df1, df2 = generate_data(
        args.table1_gnp_n,
        args.table1_gnp_ps,
        args.table2_gnp_ns,
        args.table2_gnp_p,
        args.epsilon,
        args.delta,
        args.n,
        args.p_vertex,
        args.p_edge,
    )

    df1.to_csv(table1_output_path, index=False)
    df2.to_csv(table2_output_path, index=False)
