#!/bin/bash

# Make sure this script is executable: chmod +x run_multiple_k.sh

# File to use
FILE="BPI2017O"

# List of k values you want to test
K_VALUES=(2 3 4 5 6 7 8 9 10 11)

# Loop over each k and run the Python script
for K in "${K_VALUES[@]}"; do
    echo "Running with k = $K"
    python3 adbis25_clustering.py --file "$FILE" --k "$K" --n_cpu -1
done
