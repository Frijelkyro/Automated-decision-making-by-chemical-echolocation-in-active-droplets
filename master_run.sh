#!/bin/bash

for i in {1..10}; do
    rm data/conc*.txt
    rm data/part*.txt
    python maze_cluster_script.py
    python video_maker.py
    printf -v padded "%02d" $i
    cp data/particle_trajectory.mp4 "../videos/particle_trajectory_$padded.mp4"
done