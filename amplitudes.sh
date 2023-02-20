#! /bin/sh

# Generate the three files 50.txt, 100.txt, 200.txt.
# They should be copied into Python/CosmoSim/ when generated.

exec > amplitudes.log 2>&1
time python3 Python/amplitudes.py 50 20
time python3 Python/amplitudes.py 100 20
time python3 Python/amplitudes.py 200 20
