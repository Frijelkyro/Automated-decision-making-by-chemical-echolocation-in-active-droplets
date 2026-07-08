#!/bin/bash
mkdir --parents ./output/videos; 
emission_rates=("0.5" "1" "2" "4" "8")
#Iterate through the array and apply the operations
for item in "${emission_rates[@]}"; do
    # N=1
    # mkdir --parents ./output/"$N"_particles; 
    mkdir --parents ./output/"$item"_emission_rate; 
    # sed -i -E "s/^num_particles = [0-9]+(\s*#.*)?\$/num_particles = $N # Number of particles/" maze_cluster_script.py
    sed -i -E "s/^emission_rate = [0-9.]+(\s*#.*)?\$/emission_rate = $item # droplets per second/" maze_cluster_script.py
    for i in {1..2}; do
        rm data/conc*.txt
        rm data/part*.txt
        python maze_cluster_script.py
        python video_maker.py
        printf -v padded "%02d" $i
    #    cp data/particle_trajectory.mp4 "output/videos/"$N"_particles_trajectory_$padded.mp4"
        cp data/particle_trajectory.mp4 "output/videos/"$item"_em-rate_trajectory_$padded.mp4"
    done
    # mv data/exit_times.txt ./output/"$N"_particles
    mv data/exit_times.txt ./output/"$items"_particles
done