import argparse
from pathlib import Path
from alive_progress import alive_bar
import networkx as nx
import pandas as pd
import time
import lib.intersection as intersection
import lib.epic as epic


def read_graph(file_path: any, delimiter: str | None = None) -> nx.Graph:
    G_raw = nx.read_edgelist(file_path, delimiter=delimiter, nodetype=int)
    G = nx.convert_node_labels_to_integers(G_raw)
    return G


def stopwatch(func):

    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()

        return result, end - start

    return wrapper


def generate_data(G: nx.Graph, δ: float, n: int) -> pd.DataFrame:

    i_vertex_timed = stopwatch(intersection.vertex_normalized_intersection)
    î_vertex_timed = stopwatch(epic.vertex_normalized_intersection)
    i_edge_timed = stopwatch(intersection.edge_normalized_intersection)
    î_edge_timed = stopwatch(epic.edge_normalized_intersection)

    ε_values = [0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]

    # Accuracy data
    î_vertex_mean_abs_error_values = []
    î_vertex_stddev_abs_error_values = []
    î_vertex_max_abs_error_values = []
    î_vertex_min_abs_error_values = []
    î_edge_mean_abs_error_values = []
    î_edge_stddev_abs_error_values = []
    î_edge_max_abs_error_values = []
    î_edge_min_abs_error_values = []

    # Runtime data
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

    with alive_bar(len(ε_values)) as bar:
        for ε in ε_values:
            î_vertex_abs_error_values = []
            î_vertex_max_value = 0
            î_vertex_min_value = float("inf")
            î_edge_abs_error_values = []
            î_edge_max_value = 0
            î_edge_min_value = float("inf")

            i_vertex_runtime_values = []
            î_vertex_runtime_values = []
            i_edge_runtime_values = []
            î_edge_runtime_values = []

            for _ in range(n):
                i_vertex, i_vertex_runtime = i_vertex_timed(G)
                i_vertex_runtime_values.append(i_vertex_runtime)
                print(f"Computed i_vertex in {i_vertex_runtime:.2f} seconds.")

                î_vertex, î_vertex_runtime = î_vertex_timed(
                    G, ε, δ, allow_sample_overflow=True
                )
                î_vertex_runtime_values.append(î_vertex_runtime)
                print(f"Computed î_vertex in {î_vertex_runtime:.2f} seconds.")

                i_edge, i_edge_runtime = i_edge_timed(G)
                i_edge_runtime_values.append(i_edge_runtime)
                print(f"Computed i_edge in {i_edge_runtime:.2f} seconds.")

                î_edge, î_edge_runtime = î_edge_timed(
                    G, ε, δ, allow_sample_overflow=True
                )
                î_edge_runtime_values.append(î_edge_runtime)
                print(f"Computed î_edge in {î_edge_runtime:.2f} seconds.")

                î_vertex_abs_error_values = [
                    abs(î_vertex[(u, v)] - i_vertex[(u, v)]) for u, v in i_vertex.keys()
                ]
                s_î_vertex_abs_error = pd.Series(î_vertex_abs_error_values)
                î_vertex_abs_error_values.append(s_î_vertex_abs_error.mean())
                î_vertex_max_value = max(î_vertex_max_value, s_î_vertex_abs_error.max())
                î_vertex_min_value = min(î_vertex_min_value, s_î_vertex_abs_error.min())

                î_edge_abs_error_values = [
                    abs(î_edge[(u, v)] - i_edge[(u, v)]) for u, v in i_edge.keys()
                ]
                s_î_edge_abs_error = pd.Series(î_edge_abs_error_values)
                î_edge_abs_error_values.append(s_î_edge_abs_error.mean())
                î_edge_max_value = max(î_edge_max_value, s_î_edge_abs_error.max())
                î_edge_min_value = min(î_edge_min_value, s_î_edge_abs_error.min())

            s_î_vertex_abs_error_values = pd.Series(î_vertex_abs_error_values)
            î_vertex_mean_abs_error_values.append(s_î_vertex_abs_error_values.mean())
            î_vertex_stddev_abs_error_values.append(s_î_vertex_abs_error_values.std())
            î_vertex_max_abs_error_values.append(î_vertex_max_value)
            î_vertex_min_abs_error_values.append(î_vertex_min_value)

            s_î_edge_abs_error_values = pd.Series(î_edge_abs_error_values)
            î_edge_mean_abs_error_values.append(s_î_edge_abs_error_values.mean())
            î_edge_stddev_abs_error_values.append(s_î_edge_abs_error_values.std())
            î_edge_max_abs_error_values.append(î_edge_max_value)
            î_edge_min_abs_error_values.append(î_edge_min_value)

            s_i_vertex_runtime_values = pd.Series(i_vertex_runtime_values)
            i_vertex_mean_runtime_values.append(s_i_vertex_runtime_values.mean())
            i_vertex_stddev_runtime_values.append(s_i_vertex_runtime_values.std())
            i_vertex_max_runtime_values.append(s_i_vertex_runtime_values.max())
            i_vertex_min_runtime_values.append(s_i_vertex_runtime_values.min())

            s_î_vertex_runtime_values = pd.Series(î_vertex_runtime_values)
            î_vertex_mean_runtime_values.append(s_î_vertex_runtime_values.mean())
            î_vertex_stddev_runtime_values.append(s_î_vertex_runtime_values.std())
            î_vertex_max_runtime_values.append(s_î_vertex_runtime_values.max())
            î_vertex_min_runtime_values.append(s_î_vertex_runtime_values.min())

            s_i_edge_runtime_values = pd.Series(i_edge_runtime_values)
            i_edge_mean_runtime_values.append(s_i_edge_runtime_values.mean())
            i_edge_stddev_runtime_values.append(s_i_edge_runtime_values.std())
            i_edge_max_runtime_values.append(s_i_edge_runtime_values.max())
            i_edge_min_runtime_values.append(s_i_edge_runtime_values.min())

            s_î_edge_runtime_values = pd.Series(î_edge_runtime_values)
            î_edge_mean_runtime_values.append(s_î_edge_runtime_values.mean())
            î_edge_stddev_runtime_values.append(s_î_edge_runtime_values.std())
            î_edge_max_runtime_values.append(s_î_edge_runtime_values.max())
            î_edge_min_runtime_values.append(s_î_edge_runtime_values.min())

            bar()

        return pd.DataFrame(
            {
                "epsilon": ε_values,
                "î_vertex Mean Absolute Error": î_vertex_mean_abs_error_values,
                "î_vertex StdDev Absolute Error": î_vertex_stddev_abs_error_values,
                "î_vertex Max Absolute Error": î_vertex_max_abs_error_values,
                "î_vertex Min Absolute Error": î_vertex_min_abs_error_values,
                "î_edge Mean Absolute Error": î_edge_mean_abs_error_values,
                "î_edge StdDev Absolute Error": î_edge_stddev_abs_error_values,
                "î_edge Max Absolute Error": î_edge_max_abs_error_values,
                "î_edge Min Absolute Error": î_edge_min_abs_error_values,
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "graph_input",
        type=str,
        help="Path to the graph file",
    )
    parser.add_argument(
        "--delimiter", type=str, default=None, help="Delimiter for the graph file"
    )
    parser.add_argument("--delta", type=float, default=0.1, help="Confidence parameter")
    parser.add_argument("--n", type=int, default=10, help="Number of iterations")

    args = parser.parse_args()

    G = read_graph(args.graph_input, args.delimiter)
    print(
        f"Loaded graph from {args.graph_input}. Number of nodes: {G.number_of_nodes()}. Number of edges: {G.number_of_edges()}"
    )
    graph_name = Path(args.graph_input).stem
    Path("./results").mkdir(parents=True, exist_ok=True)
    output_path = Path(
        f"./results/normalized_intersection_data_{graph_name}_delta_{args.delta}_n_{args.n}.csv"
    )

    df = generate_data(G, args.delta, args.n)
    df.to_csv(output_path, index=False)
