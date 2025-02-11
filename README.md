# EPIC

> This repository contains the source code for EPIC and all scripts used to generate the data used in article Efficient Approximations of Neighborhood Intersection in Large Graphs via Sampling (link omitted due to double-blind peer review).

EPIC (Epsilon-Delta Probabilistic Intersection Calculator) consists of four algorithms that approximate the _neighborhood intersection_ for all pairs of vertices is a graph.
| Algorithm | Neighborhood Intersection type | Error bound type |
|:-------------------:|:------------------------------:|:----------------:|
| `intersection_vertex_sampling` | Default | Multiplicative |
| `intersection_edge_sampling` | Default | Multiplicative |
| `vertex_normalized_intersection` | Vertex-normalized | Additive |
| `edge_normalized_intersection` | Edge-normalized | Additive |

Each algorithm guarantees an additive/multiplicative error bound $\epsilon$ with probability at least $1 - \delta$. Go to `lib/epic.py` to see the implementation of the algorithms.
