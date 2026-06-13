#! /bin/bash
for i in $(seq 1 10); do rm data/part* && rm data/conc* && python maze_cluster_script.py; done