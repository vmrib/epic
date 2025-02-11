#!/bin/sh

# Generate all data used in the paper, using the same graphs and parameters as in the paper.

echo "Generating data for Tables 1 and 2..."
python3 generate_intersection_data.py --epsilon 0.1 --delta 0.1 --n 10 --p_vertex 0.7 --p_edge 0.04 --table1_gnp_n 1000 --table1_gnp_ps 0.1 0.3 0.5 0.7 0.9 --table2_gnp_ns 500 1000 2000 4000 --table2_gnp_p 0.5

echo -e "\nGenerating data for Table 3..."
python3 generate_dataset_details.py graphs/facebook_new_sites_edges.csv graphs/deezer_RO_edges.csv graphs/loc-brightkite_edges.txt

echo -e "\nGenerating data for Figure 2..."
python3 generate_normalized_intersection_data.py graphs/deezer_RO_edges.csv --delimiter , --delta 0.1 --n 10

echo -e "\nGenerating data for Figure 3..."
python3 generate_normalized_intersection_data.py graphs/facebook_new_sites_edges.csv --delimiter , --delta 0.1 --n 10

echo -e "\nGenerating data for Figure 4..."
python3 generate_normalized_intersection_data.py graphs/loc-brightkite_edges.txt --delta 0.1 --n 10

echo -e "\nGenerating data for Figure 5..."
python generate_normalized_intersection_scalability_data.py --epsilon 0.05 --delta 0.1 --n 10 --barabasi_m 10 --barabasi_ns 10000 20000 30000 40000 50000 60000 80000 90000 100000

echo -e "\nGenerating data for Figures 6 and 7..."
python3 generate_heatmaps_data.py --epsilon 0.1 --delta 0.1 --n 10 --gnp_ns 1000 2000 3000 4000 5000 --gnp_ps 0.05 0.1 0.3 0.5 0.7 0.9

echo -e "\nData generation complete. All data is stored in the './results' directory.
Table 1   -> ./results/intersection_data_table_1_*.csv
Table 2   -> ./results/intersection_data_table_2_*.csv
Table 3   -> ./results/dataset_details.csv
Figure 2  -> ./results/normalized_intersection_data_deezer_RO_edges_*.csv
Figure 3  -> ./results/normalized_intersection_data_facebook_new_sites_edges_*.csv
Figure 4  -> ./results/normalized_intersection_data_loc-brightkite_edges_*.csv
Figure 5  -> ./results/normalized_intersection_scalability_data_*.csv
Figures 6 -> ./results/heat_map_vertex_sampling_*.csv
Figures 7 -> ./results/heat_map_edge_sampling_*.csv"