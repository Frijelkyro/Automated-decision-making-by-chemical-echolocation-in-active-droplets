#!/bin/bash
mkdir --parents ./output/videos; 
N=1
mkdir --parents ./output/"$N"_particles; 
sed -i -E "s/^num_particles = [0-9]+(\s*#.*)?\$/num_particles = $N # Number of particles/" maze_cluster_script.py
for i in {1..50}; do
    rm data/conc*.txt
    rm data/part*.txt
    python maze_cluster_script.py
    python video_maker.py
    printf -v padded "%02d" $i
    cp data/particle_trajectory.mp4 "output/videos/"$N"_particles_trajectory_$padded.mp4"
done
mv data/exit_times.txt ./output/"$N"_particles
N=2
mkdir --parents ./output/"$N"_particles/videos;  
sed -i -E "s/^num_particles = [0-9]+(\s*#.*)?\$/num_particles = $N # Number of particles/" maze_cluster_script.py
for i in {1..50}; do
    rm data/conc*.txt
    rm data/part*.txt
    python maze_cluster_script.py
    python video_maker.py
    printf -v padded "%02d" $i
    cp data/particle_trajectory.mp4 "output/videos/"$N"_particles_trajectory_$padded.mp4"
done
mv data/exit_times.txt ./output/"$N"_particles
N=4
mkdir --parents ./output/"$N"_particles/videos;  
sed -i -E "s/^num_particles = [0-9]+(\s*#.*)?\$/num_particles = $N # Number of particles/" maze_cluster_script.py
for i in {1..50}; do
    rm data/conc*.txt
    rm data/part*.txt
    python maze_cluster_script.py
    python video_maker.py
    printf -v padded "%02d" $i
    cp data/particle_trajectory.mp4 "output/videos/"$N"_particles_trajectory_$padded.mp4"
done
mv data/exit_times.txt ./output/"$N"_particles
N=10
mkdir --parents ./output/"$N"_particles/videos;  
sed -i -E "s/^num_particles = [0-9]+(\s*#.*)?\$/num_particles = $N # Number of particles/" maze_cluster_script.py
for i in {1..50}; do
    rm data/conc*.txt
    rm data/part*.txt
    python maze_cluster_script.py
    python video_maker.py
    printf -v padded "%02d" $i
    cp data/particle_trajectory.mp4 "output/videos/"$N"_particles_trajectory_$padded.mp4"
done
mv data/exit_times.txt ./output/"$N"_particles