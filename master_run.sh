#!/bin/bash

# Ensure base video directory exists
mkdir --parents ./output/videos

# Outer loop for particle counts
# for N in 1 2 4 10; do
#     echo "=== Running simulation for N = $N particles ==="
#     
#     # Create output directories
#     mkdir --parents ./output/"$N"_particles
#     
#     # Update the python script with the current particle count
#     sed -i -E "s/^num_particles = [0-9]+(\s*#.*)?\$/num_particles = $N # Number of particles/" maze_cluster_script.py
#     
#     # Inner loop for iterations
#     for i in {1..50}; do
#         # Clean up old data files safely
#         rm -f data/conc*.txt
#         rm -f data/part*.txt
#         
#         # Run simulation and video generator
#         python maze_cluster_script.py
#         if [ $i -le 5 ]; then python video_maker.py; fi
#         
#         # Format the frame padding (e.g., 01, 02...)
#         printf -v padded "%02d" $i
#         
#         # Copy output trajectory video
#         cp data/particle_trajectory.mp4 "output/videos/${N}_particles_trajectory_$padded.mp4"
#     done
#     
#     # Move the exit summary log to its final folder
#     mv data/exit_times.txt ./output/"$N"_particles/
# done

echo "All particle number simulations complete!"

# =====================================================================
# SECTION 2: Varying Emission Rate (With Dynamic Particle Scaling)
# =====================================================================

# Outer loop for emission rates
for ER in 0.25 0.5 1 2 4 10; do
    # Calculate corresponding particle count (100 * emission_rate) using bc for floats
    CALCULATED_PARTICLES=$(echo "$ER * 100" | bc | cut -d'.' -f1)
    
    echo "=== Running simulation: emission_rate = $ER ($CALCULATED_PARTICLES particles) ==="
    
    # Create distinct output directories
    mkdir --parents ./output/"${ER}"_emission_rate
    
    # Update BOTH num_particles and emission_rate in the python script
    sed -i -E "s/^num_particles = [0-9]+(\s*#.*)?\$/num_particles = $CALCULATED_PARTICLES # Number of particles/" maze_cluster_script.py
    sed -i -E "s/^emission_rate = [0-9.]+(\s*#.*)?\$/emission_rate = $ER # droplets per second/" maze_cluster_script.py
    
    # Inner loop for iterations
    for i in {1..50}; do
        rm -f data/conc*.txt
        rm -f data/part*.txt
        
        python maze_cluster_script.py
        if [ $i -le 5 ]; then python video_maker.py; fi
        
        printf -v padded "%02d" $i
        cp data/particle_trajectory.mp4 "output/videos/${ER}_emission_trajectory_$padded.mp4"
    done
    
    mv data/exit_times.txt ./output/"${ER}"_emission_rate/
done

echo "All emission rate simulations complete!"