#!/bin/bash

# =====================================================================
# REUSABLE RECOVERY FUNCTION (Handles Crash Logs & Data Extraction)
# =====================================================================
process_crash_recovery() {
    local target_log_dir="$1"   # Argument 1: Path to the crash log destination
    local output_file="data/exit_times.txt"

    echo "=== Simulation failed! Running fallback data retrieval ==="

    # 1. Backup crash logs and run video maker
    mkdir -p "$target_log_dir"
    cp -r data/* "$target_log_dir/"
    python video_maker.py

    # 2. Extract TIME_INCREMENT float from data/param.txt
    local time_increment
    time_increment=$(awk '/^dt:[[:space:]]*/ {print $2}' data/param.txt)

    # 3. Find the maximum timestep file numerically
    local max_timestep_file
    max_timestep_file=$(ls data/part_*.txt 2>/dev/null | sed 's/[^0-9]//g' | sort -n | tail -n 1)
    
    if [ -z "$max_timestep_file" ]; then
        echo "Error: No data files found to process."
        return 1
    fi
    
    # 4. Find ordered array of particle IDs that are "nan" in the final step
    local nan_particles
    nan_particles=($(awk '$1 ~ /^[0-9]{1,4}$/ && $2 == "nan" {print $1}' "data/part_${max_timestep_file}.txt" | sort -n))

    # Initialize exit_times file with header
    echo "ExitTime Beta JobID ParticleID" > "$output_file"

    # Get reverse-sorted timesteps for faster backward scanning
    local all_timesteps
    all_timesteps=($(ls data/part_*.txt 2>/dev/null | sed 's/[^0-9]//g' | sort -rn))

    # 5. Scan backwards to find when each nan particle exited
    for pid in "${nan_particles[@]}"; do
        local exit_timestep=""
        
        for ts in "${all_timesteps[@]}"; do
            # Simple, fast match query
            local status
            status=$(awk -v id="$pid" '$1 == id { print ($2 != "nan" && $2 != "") ? "MATCH" : "NAN"; exit }' "data/part_${ts}.txt")
            
            if [ "$status" = "MATCH" ]; then
                exit_timestep="$ts"
                break
            fi
        done

        if [ -n "$exit_timestep" ]; then
            local exit_time
            exit_time=$(echo "$exit_timestep * $time_increment" | bc -l)
            printf "%.3f -1 1 %d\n" "$exit_time" "$pid" >> "$output_file"
        else
            printf "0.0 -1 1 %d\n" "$pid" >> "$output_file"
        fi
    done
    echo "Fallback data recovery complete."
}

# Ensure base video directory exists
mkdir --parents ./output/videos


# =====================================================================
# SECTION 1: Varying Particle Counts
# =====================================================================
for N in 1; do
    echo "=== Running simulation for N = $N particles ==="
    mkdir --parents ./output/"$N"_particles
    
    # Update the python script with the current particle count
    sed -i -E "s/^num_particles = [0-9]+(\s*#.*)?\$/num_particles = $N # Number of particles/" maze_cluster_script.py
    
    for i in {1..3}; do
        printf -v padded "%02d" $i
        rm -f data/conc*.txt data/part*.txt
        
        python maze_cluster_script.py
        
        if [ $? -ne 0 ]; then
            # Calling the recovery function for Section 1
            process_crash_recovery "./output/${N}_particles/crash_logs/${padded}/data"
        fi
        if [ $i -le 5 ]; then
            python video_maker.py
            cp data/particle_trajectory.mp4 "output/videos/${N}_particles_trajectory_$padded.mp4"
        fi
        if [ $i -eq 25 ]; then
            echo "interation number: 25"
        fi
    done
    
    mv data/exit_times.txt ./output/"$N"_particles/
done

echo "All particle number simulations complete!"


# =====================================================================
# SECTION 2: Varying Emission Rate
# =====================================================================
for ER in 0.5 1 2 4 10; do
    CALCULATED_PARTICLES=$(echo "$ER * 100" | bc | cut -d'.' -f1)
    echo "=== Running simulation: emission_rate = $ER ($CALCULATED_PARTICLES particles) ==="
    mkdir --parents ./output/"${ER}"_emission_rate
    
    sed -i -E "s/^num_particles = [0-9]+(\s*#.*)?\$/num_particles = $CALCULATED_PARTICLES # Number of particles/" maze_cluster_script.py
    sed -i -E "s/^emission_rate = [0-9.]+(\s*#.*)?\$/emission_rate = $ER # droplets per second/" maze_cluster_script.py
    
    for i in {1..3}; do
        printf -v padded "%02d" $i
        rm -f data/conc*.txt data/part*.txt
        
        python maze_cluster_script.py

        if [ $? -ne 0 ]; then
            # Calling the same recovery function for Section 2
            process_crash_recovery "./output/${ER}_emission_rate/crash_logs/${padded}/data"
        fi
    done
    
    mv data/exit_times.txt ./output/"${ER}"_emission_rate/
done

echo "All emission rate simulations complete!"