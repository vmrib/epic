
# EPIC

This repository contains the source code for EPIC and all scripts used to generate the data used in article Efficient Approximations of Neighborhood Intersection in Large Graphs via Sampling (link omitted due to double-blind peer review).

EPIC (**E**fficient **P**robabilistic Neighborhood **I**ntersection **C**alculator) consists of four algorithms that approximate the _neighborhood intersection_ for all pairs of vertices is a graph.
| Algorithm | Neighborhood Intersection Type | Error bound type |
|:-------------------:|:------------------------------:|:----------------:|
| `intersection_vertex_sampling` | Default | Multiplicative |
| `intersection_edge_sampling` | Default | Multiplicative |
| `vertex_normalized_intersection` | Vertex-normalized | Additive |
| `edge_normalized_intersection` | Edge-normalized | Additive |

Each algorithm guarantees an additive/multiplicative error bound $\epsilon$ with probability at least $1 - \delta$. Go to `lib/epic.py` for the algorithm implementations.

## Setup

You'll need Git and Python >=3.10 installed on your computer. From you command line:

```bash
# Clone this repository (link omitted due to double-blind peer review)
git clone --depth 1 <this project url>

# Go into the repository
cd epic

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Reproducibility

To generate all data used in the paper, with the same graphs and parameters:

```bash
source generate_all_paper_data.sh
```

You can also run individual scripts with custom options:

- `generate_dataset_details.py`: Generates data containg the number of nodes, number of edges, maximum degree, and non-empty intersections ratio for multiple graphs.
- `generate_heatmaps_data.py`: Generates data for heatmaps visualizing the Minimum Threshold p for vertex and edge sampling strategies on GNP graphs.
- `generate_intersection_data.py`: Generates data for comparing vertex and edge sampling strategies on GNP graphs.  The data includes statistical measures (mean, standard deviation, minimum, and maximum) for runtime and the success rate of approximations for both strategies.
- `generate_normalized_intersection_data`: Generates data for analyzing the performance of normalized intersection algorithms. The data includes statistical measures (mean, standard deviation, max, min) for the average absolute error and runtime for both vertex-normalized and edge-normalized intersection algorithms as a function of the accuracy parameter $\epsilon$.
- `generate_normalized_intersection_scalability_data`: Generates data for analyzing the scalability of normalized intersection algorithms (vertex- and edge-normalized). The data includes statistical measures (mean, standard deviation, max, min) of the runtime of the exact and approximate algorithms.

| Script                                              | Output                                                   | Where it is used in the paper |
|-----------------------------------------------------|----------------------------------------------------------|-------------------------------|
| `generate_dataset_details.py`                       | `results/dataset_details.csv`                            | Table 3                       |
| `generate_heatmaps_data.py`                         | `results/heat_map_*.csv`                                 | Figures 6 and 7               |
| `generate_intersection_data.py`                     | `results/intersection_data_*.csv`                        | Tables 1 and 2                |
| `generate_normalized_intersection_data`             | `results/normalized_intersection_data_*.csv`             | Figures 2, 3 and 4            |
| `generate_normalized_intersection_scalability_data` | `results/normalized_intersection_scalability_data_*.csv` | Figure 5                      |

To see all the custom options for a script, use the `--help` flag. For example:
```bash
# Show custom options for generating the heatmaps
python3 generate_heatmaps_data.py --help

# Generate heatmaps with custom options
python3 generate_heatmaps_data.py --epsilon 0.1 --delta 0.1 --n 42 --gnp_ns 500 1000 1500 2000 4000 --gnp_ps 0.1 0.2 0.3 0.4 0.5 0.7 0.9
```