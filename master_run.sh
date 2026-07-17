#!/bin/bash

# =====================================================================
# REUSABLE RECOVERY FUNCTION (Handles Crash Logs & Data Extraction)
# =====================================================================
# TODO Fix Video saving Issue
process_crash_recovery() {
    local target_log_dir="$1"   # Argument 1: Path to the crash log destination
    local output_file="data/exit_times.txt"

    echo "=== Simulation failed! Running fallback data retrieval ==="

    # 1. Backup crash logs and run video maker
    mkdir -p "$target_log_dir"
    cp -r data/* "$target_log_dir/"

    # 2. Extract TIME_INCREMENT float from data/param.txt
    local time_increment
    time_increment=$(awk '/^dt:[[:space:]]*/ {print $2}' data/param.txt)

    # 3. Extract all numbers, sort numerically for ascending and descending arrays
    local all_timesteps_asc=($(ls data/part_*.txt 2>/dev/null | sed 's/[^0-9]//g' | sort -n))
    local all_timesteps_desc=($(ls data/part_*.txt 2>/dev/null | sed 's/[^0-9]//g' | sort -rn))
    
    if [ ${#all_timesteps_asc[@]} -eq 0 ]; then
        echo "Error: No data files found to process."
        return 1
    fi
    
    local first_timestep_file="${all_timesteps_asc[0]}"
    local max_timestep_file="${all_timesteps_desc[0]}"
    
    # 4. Get the un-injected "baseline" coordinates (last particle's position in step 0)
    local baseline_coords=($(awk '$1 ~ /^[0-9]+$/ {x=$2; y=$3} END {print x, y}' "data/part_${first_timestep_file}.txt"))
    local base_x="${baseline_coords[0]}"
    local base_y="${baseline_coords[1]}"
    
    # 5. Find ordered array of particle IDs that are "nan" in the final step
    local nan_particles=($(awk '$1 ~ /^[0-9]{1,4}$/ && $2 == "nan" {print $1}' "data/part_${max_timestep_file}.txt" | sort -n))

    if [[ ! -f "$output_file" ]]; then
        echo "ExitTime Beta JobID ParticleID" > "$output_file"
    fi

    # 6. Process each exited particle
    for pid in "${nan_particles[@]}"; do
        local exit_timestep=""
        local start_timestep=""
        
        # --- A. Scan BACKWARDS to find the last known position before exiting ---
        for ts in "${all_timesteps_desc[@]}"; do
            local status
            status=$(awk -v id="$pid" '$1 == id { print ($2 != "nan" && $2 != "") ? "MATCH" : "NAN"; exit }' "data/part_${ts}.txt")
            
            if [ "$status" = "MATCH" ]; then
                exit_timestep="$ts"
                break
            fi
        done

        if [ -n "$exit_timestep" ]; then
            # --- B. Scan FORWARDS to find the first active movement ---
            for ts in "${all_timesteps_asc[@]}"; do
                local moved
                moved=$(awk -v id="$pid" -v bx="$base_x" -v by="$base_y" '
                    $1 == id {
                        # A particle is active if it is not nan AND its coordinates do not match the baseline
                        if ($2 != "nan" && $2 != "" && ($2 != bx || $3 != by)) {
                            print "MOVED"
                        } else {
                            print "WAIT"
                        }
                        exit
                    }' "data/part_${ts}.txt")
                
                if [ "$moved" = "MOVED" ]; then
                    start_timestep="$ts"
                    break
                fi
            done
            
            # Failsafe: if no movement was detected, default to the very first file
            if [ -z "$start_timestep" ]; then
                start_timestep="$first_timestep_file"
            fi

            # --- C. Calculate the delta and append ---
            local exit_time
            exit_time=$(echo "($exit_timestep - $start_timestep) * $time_increment" | bc -l)
            printf "%.3f -8 1 %d\n" "$exit_time" "$pid" >> "$output_file"
        else
            printf "-1 -8 1 %d\n" "$pid" >> "$output_file"
        fi
    done

    echo "Fallback data recovery complete."
}

# Ensure base video directory exists
mkdir --parents ./output/videos


# =====================================================================
# SECTION 1: Varying Particle Counts
# =====================================================================
for N in 1 2 4; do
    echo "=== Running simulation for N = $N particles ==="
    
    # FIXED: Added instant_release to path to match the cp/mv commands below
    mkdir --parents "./output/instant_release/${N}_particles"

    # FIXED: Corrected sed syntax with .* and trailing /
    sed -i -E "s/^drops_added_incremental =.*/drops_added_incremental = False/" maze_cluster_script.py
        
    case "$N" in
        10)   DT=0.05 ;;
        *)    DT=0.25 ;;
    esac

    # Update the python script with the current particle count
    sed -i -E "s/^num_particles = [0-9]+(\s*#.*)?\$/num_particles = $N # Number of particles/" maze_cluster_script.py
    sed -i -E "s/^dt = .*/dt = $DT * 10 ** (-3)  # time step size/" maze_cluster_script.py
    grep "dt = " maze_cluster_script.py

    for i in {1..50}; do
        printf -v padded "%02d" $i
        rm -f ./data/conc*.txt ./data/part*.txt
        
        python maze_cluster_script.py
        
        if [ $? -ne 0 ]; then
            # Calling the recovery function for Section 1
            process_crash_recovery "./output/instant_release/${N}_particles/crash_logs/${padded}/data"
        fi
        if [ $i -le 5 ]; then
            python video_maker.py
            cp ./data/particle_trajectory.mp4 "./output/instant_release/${N}_particles/${N}_particles_trajectory_${padded}.mp4"
        fi
        if [ $i -eq 25 ]; then
            echo "interation number: 25"
        fi
    done

    mkdir --parents output/xbucket
    echo "# exit_times.txt, Simultaneuos Release, N: ${N}"
    cat ./data/exit_times.txt >> output/xbucket/recover_data.txt
    mv ./data/exit_times.txt "./output/instant_release/${N}_particles/"
done

echo "All particle number simulations complete!"


# =====================================================================
# SECTION 2: Varying Emission Rate
# =====================================================================
# FIXED: Corrected sed syntax with .* and trailing /
sed -i -E "s/^drops_added_incremental =.*/drops_added_incremental = True/" maze_cluster_script.py

for ER in 0.25 0.50 1 2 4 10; do
    # Calculate particles
    CALCULATED_PARTICLES=$(echo "$ER * 100" | bc | cut -d'.' -f1)
    echo "=== Running simulation: emission_rate = $ER ($CALCULATED_PARTICLES particles) ==="
    mkdir --parents "./output/${ER}_emission_rate"

    # Use a case statement instead of integer comparison for floats
    case "$ER" in
        0.25) DT=1.0 ;;
        0.5)  DT=0.5 ;;
        1)    DT=0.25 ;;
        2)    DT=0.12 ;;
        4)    DT=0.08 ;; # 0.25
        10)   DT=0.04 ;; # 0.10
        *)    DT=0.25 ;; # Default fallback just in case
    esac
    
    # Update Python script
    sed -i -E "s/^num_particles = [0-9]+(\s*#.*)?\$/num_particles = $CALCULATED_PARTICLES # Number of particles/" maze_cluster_script.py
    sed -i -E "s/^emission_rate = [0-9.]+(\s*#.*)?\$/emission_rate = $ER # droplets per second/" maze_cluster_script.py
    sed -i -E "s/^dt = .*/dt = $DT * 10 ** (-3)  # time step size/" maze_cluster_script.py
    grep "dt =" maze_cluster_script.py 
    
    # Safely remove exit times without throwing an error if it doesn't exist
    rm -f data/exit_times.txt

    for i in {1..50}; do
        printf -v padded "%02d" $i
        rm -f data/conc*.txt data/part*.txt
        
        python maze_cluster_script.py
        exit_status=$?

        if [ $exit_status -ne 0 ]; then
            process_crash_recovery "./output/${ER}_emission_rate/crash_logs/${padded}/data"
        fi

        if [ $i -eq 1 ]; then
            python video_maker.py
            cp ./data/particle_trajectory.mp4 "./output/${ER}_emission_rate/${ER}_emission_trajectory_${padded}.mp4"
        fi
    done
    
    if [ -f ./data/exit_times.txt ]; then
        cp ./data/exit_times.txt "./output/${ER}_emission_rate/"
    else
        echo "Warning: exit_times.txt not found for ER=$ER"
    fi
done

echo "All emission rate simulations complete!"