#!/bin/bash

# Ensure base video directory exists
mkdir --parents ./output/videos

Outer loop for particle counts
for N in 1; do
#for N in 1 2 4 10; do
    echo "=== Running simulation for N = $N particles ==="
    
    # Create output directories
    mkdir --parents ./output/"$N"_particles
    
    # Update the python script with the current particle count
    sed -i -E "s/^num_particles = [0-9]+(\s*#.*)?\$/num_particles = $N # Number of particles/" maze_cluster_script.py
    
    # Inner loop for iterations
    for i in {1..50}; do
        # Clean up old data files safely
        rm -f data/conc*.txt
        rm -f data/part*.txt
        
        # Run simulation and video generator
        python maze_cluster_script.py
        
        if [ $i -le 5 ]; then
            python video_maker.py; fi
            printf -v padded "%02d" $i
            cp data/particle_trajectory.mp4 "output/videos/${N}_particles_trajectory_$padded.mp4"
        fi
    done
    
    # Move the exit summary log to its final folder
    mv data/exit_times.txt ./output/"$N"_particles/
done

echo "All particle number simulations complete!"
# =====================================================================
# SECTION 2: Varying Emission Rate (With Dynamic Particle Scaling)
# =====================================================================

# Outer loop for emission rates
for ER in 0.5 1 2 4 10; do
    # Calculate corresponding particle count (100 * emission_rate) using bc for floats
    CALCULATED_PARTICLES=$(echo "$ER * 100" | bc | cut -d'.' -f1)
    
    echo "=== Running simulation: emission_rate = $ER ($CALCULATED_PARTICLES particles) ==="
    
    # Create distinct output directories
    mkdir --parents ./output/"${ER}"_emission_rate
    
    # Update BOTH num_particles and emission_rate in the python script
    sed -i -E "s/^num_particles = [0-9]+(\s*#.*)?\$/num_particles = $CALCULATED_PARTICLES # Number of particles/" maze_cluster_script.py
    sed -i -E "s/^emission_rate = [0-9.]+(\s*#.*)?\$/emission_rate = $ER # droplets per second/" maze_cluster_script.py
    
    # Inner loop for iterations
    for i in {1..25}; do
        rm -f data/conc*.txt
        rm -f data/part*.txt
        
        python maze_cluster_script.py

    # Check if the simulation failed (exit code is non-zero)
    if [ $? -ne 0 ]; then
        # 2. Backup crash logs and run video maker
        mkdir -p "./output/${ER}_emission_rate/crash_logs/${padded}/data"
        cp -r data/* "./output/${ER}_emission_rate/crash_logs/${padded}/data/"
        python video_maker.py

        # 3. Process the data manually since python crashed
        # Extract TIME_INCREMENT float from data/param.txt
        TIME_INCREMENT=$(awk '/^dt:[[:space:]]*/ {print $2}' data/param.txt)

        # Find the maximum timestep file numerically
        MAX_TIMESTEP_FILE=$(ls data/part_*.txt 2>/dev/null | sed 's/[^0-9]//g' | sort -n | tail -n 1)
        
        if [ -z "$MAX_TIMESTEP_FILE" ]; then
            echo "Error: No data files found to process."
            exit 1
        fi
        
        # Get ordered array of particle IDs that are "nan" in the final step
        nan_particles=($(awk '$1 ~ /^[0-9]{1,4}$/ && $2 == "nan" {print $1}' "data/part_${MAX_TIMESTEP_FILE}.txt" | sort -n))

        # Initialize data/exit_times.txt file with header
        OUTPUT_FILE="data/exit_times.txt"
        echo "ExitTime Beta JobID ParticleID" > "$OUTPUT_FILE"

        # Get reverse-sorted timesteps for faster backward scanning
        all_timesteps=($(ls data/part_*.txt 2>/dev/null | sed 's/[^0-9]//g' | sort -rn))

        # 4. Scan backwards to find when each nan particle exited
        for pid in "${nan_particles[@]}"; do
            EXIT_TIMESTEP=""
            
            for ts in "${all_timesteps[@]}"; do
                # Robust AWK check: Strips hidden white spaces around columns
                # Exits 0 (Success) the exact microsecond it hits actual numbers instead of nan
                if awk -v id="$pid" 'BEGIN {found=0} {gsub(/^[ \t]+|[ \t]+$/, ""); if ($1 == id && $2 != "nan" && $2 != "") {found=1; exit 0}} END {if (found==0) exit 1}' "data/part_${ts}.txt"; then
                    EXIT_TIMESTEP="$ts"
                    break
                fi
            done

            if [ -n "$EXIT_TIMESTEP" ]; then
                # Calculate floating point time via bc
                EXIT_TIME=$(echo "$EXIT_TIMESTEP * $TIME_INCREMENT" | bc -l)
                printf "%.3f -1 1 %d\n" "$EXIT_TIME" "$pid" >> "$OUTPUT_FILE"
            else
                printf "0.0 -1 1 %d\n" "$pid" >> "$OUTPUT_FILE"
            fi
        done
    fi
    done
    
    mv data/exit_times.txt ./output/"${ER}"_emission_rate/
done

echo "All emission rate simulations complete!"