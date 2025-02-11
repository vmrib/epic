# Generate all data used in the paper, using the same graphs and parameters as in the paper.

python3 generate_dataset_details.py graphs/facebook_new_sites_edges.csv graphs/deezer_RO_edges.csv graphs/loc-brightkite_edges.txt

python3 generate_heatmaps_data.py --epsilon 0.1 --delta 0.1 --n 10 --gnp_ns 1000 2000 3000 4000 5000 --gnp_ps 0.05 0.1 0.3 0.5 0.7 0.9

python3 generate_intersection_data.py --epsilon 0.1 --delta 0.1 --n 10 --p_vertex 0.7 --p_edge 0.04 --table1_gnp_n 1000 --table1_gnp_ps 0.1 0.3 0.5 0.7 0.9 --table2_gnp_ns 500 1000 2000 4000 --table2_gnp_p 0.5

python3 generate_normalized_intersection_data.py graphs/facebook_new_sites_edges.csv --delimiter , --delta 0.1 --n 10

python3 generate_normalized_intersection_data.py graphs/deezer_RO_edges.csv --delimiter , --delta 0.1 --n 10

python3 generate_normalized_intersection_data.py graphs/loc-brightkite_edges.txt --delta 0.1 --n 10

python generate_normalized_intersection_scalability_data.py --epsilon 0.05 --delta 0.1 --n 10 --barabasi_m 10 --barabasi_ns 10000 20000 30000 40000 50000 60000 80000 90000 100000