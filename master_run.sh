#!/bin/bash

# =====================================================================
# REUSABLE RECOVERY FUNCTION (Handles Crash Logs & Data Extraction)
# =====================================================================
# TODO Fix Video saving Issue
log_and_exit_times_recovery() {
    local target_log_dir="$1"   # Argument 1: Path to the crash log destination

    echo "=== Simulation failed! Running fallback data retrieval ==="

    mkdir -p "$target_log_dir"
    cp -r data/* "$target_log_dir/"
    python recovery.py
    echo "Fallback data recovery complete."
}

# Ensure base video directory exists
mkdir --parents ./output/videos


## =====================================================================
## SECTION 1: Varying Particle Counts
## =====================================================================
#for N in 1 2 4; do
#    echo "=== Running simulation for N = $N particles ==="
#    
#    # FIXED: Added instant_release to path to match the cp/mv commands below
#    mkdir --parents "./output/instant_release/${N}_particles"
#
#    # FIXED: Corrected sed syntax with .* and trailing /
#    sed -i -E "s/^drops_added_incremental =.*/drops_added_incremental = False/" maze_cluster_script.py
#        
#    case "$N" in
#        10)   DT=0.05 ;;
#        *)    DT=0.25 ;;
#    esac
#
#    # Update the python script with the current particle count
#    sed -i -E "s/^num_particles = [0-9]+(\s*#.*)?\$/num_particles = $N # Number of particles/" maze_cluster_script.py
#    sed -i -E "s/^dt = .*/dt = $DT * 10 ** (-3)  # time step size/" maze_cluster_script.py
#    grep "dt = " maze_cluster_script.py
#
#    for i in {1..50}; do
#        printf -v padded "%02d" $i
#        rm -f ./data/conc*.txt ./data/part*.txt
#        
#        python maze_cluster_script.py
#        
#        if [ $? -ne 0 ]; then
#            # Calling the recovery function for Section 1
#            log_and_exit_times_recovery "./output/instant_release/${N}_particles/crash_logs/${padded}/data"
#        fi
#        if [ $i -le 5 ]; then
#            python video_maker.py
#            cp ./data/particle_trajectory.mp4 "./output/instant_release/${N}_particles/${N}_particles_trajectory_${padded}.mp4"
#        fi
#        if [ $i -eq 25 ]; then
#            echo "interation number: 25"
#        fi
#    done
#
#    mkdir --parents output/xbucket
#    echo "# exit_times.txt, Simultaneuos Release, N: ${N}"
#    cat ./data/exit_times.txt >> output/xbucket/recover_data.txt
#    mv ./data/exit_times.txt "./output/instant_release/${N}_particles/"
#done
#
#echo "All particle number simulations complete!"
#
#
## =====================================================================
## SECTION 2: Varying Emission Rate
## =====================================================================
## FIXED: Corrected sed syntax with .* and trailing /
#sed -i -E "s/^drops_added_incremental =.*/drops_added_incremental = True/" maze_cluster_script.py
#
#for ER in 0.25 0.50 1 2 4 10; do
#    # Calculate particles
#    CALCULATED_PARTICLES=$(echo "$ER * 100" | bc | cut -d'.' -f1)
#    echo "=== Running simulation: emission_rate = $ER ($CALCULATED_PARTICLES particles) ==="
#    mkdir --parents "./output/dripping_release/${ER}_emission_rate" #TEST THIS
#
#    # Use a case statement instead of integer comparison for floats
#    case "$ER" in
#        0.25) DT=1.0 ;;
#        0.5)  DT=0.5 ;;
#        1)    DT=0.25 ;;
#        2)    DT=0.12 ;;
#        4)    DT=0.08 ;; # 0.25
#        10)   DT=0.04 ;; # 0.10
#        *)    DT=0.25 ;; # Default fallback just in case
#    esac
#    
#    # Update Python script
#    sed -i -E "s/^num_particles = [0-9]+(\s*#.*)?\$/num_particles = $CALCULATED_PARTICLES # Number of particles/" maze_cluster_script.py
#    sed -i -E "s/^emission_rate = [0-9.]+(\s*#.*)?\$/emission_rate = $ER # droplets per second/" maze_cluster_script.py
#    sed -i -E "s/^dt = .*/dt = $DT * 10 ** (-3)  # time step size/" maze_cluster_script.py
#    grep "dt =" maze_cluster_script.py 
#    
#    # Safely remove exit times without throwing an error if it doesn't exist
#    rm -f data/exit_times.txt
#
#    for i in {1..50}; do
#        printf -v padded "%02d" $i
#        rm -f data/conc*.txt data/part*.txt
#        
#        python maze_cluster_script.py
#        exit_status=$?
#
#        if [ $exit_status -ne 0 ]; then
#            log_and_exit_times_recovery "./output/${ER}_emission_rate/crash_logs/${padded}/data"
#        fi
#
#        if [ $i -eq 1 ]; then
#            python video_maker.py
#            cp ./data/particle_trajectory.mp4 "./output/${ER}_emission_rate/${ER}_emission_trajectory_${padded}.mp4"
#        fi
#    done
#    
#    if [ -f ./data/exit_times.txt ]; then
#        cp ./data/exit_times.txt "./output/${ER}_emission_rate/"
#    else
#        echo "Warning: exit_times.txt not found for ER=$ER"
#    fi
#done
#
#echo "All emission rate simulations complete!"
#
#
## Section 3
#
## for REAPER_TIMER in 64.0 32.0 16.0 8.0 4.0 2.0 1.0 0.5 0.25 0.1 0.05 0 ; do
# for REAPER_TIMER in 64.0 32.0 16.0 8.0 4.0 2.0 1.0 0.5 0.25 0.1 0.05 0 ; do
# for REAPER_TIMER in 48.0 24.0 12.0 6.0 3.0 1.6 0.8 0.4 0.2 0.1 0.01; do
for REAPER_TIMER in 12.1; do  #0.4 0.2 0.1 0.01; do
    ER=3.5
    DT=0.9
    CALCULATED_PARTICLES=$(echo "$ER * 400" | bc | cut -d'.' -f1)
    DESIRED_TIME=1000
    sed -i -E "s/^num_particles = [0-9]+(\s*#.*)?\$/num_particles = $CALCULATED_PARTICLES # Number of particles/" maze_cluster_script.py
    sed -i -E "s/^emission_rate = [0-9.]+(\s*#.*)?\$/emission_rate = $ER # droplets per second/" maze_cluster_script.py
    sed -i -E "s/^dt = .*/dt = $DT * 10 ** (-3)  # time step size/" maze_cluster_script.py  
    sed -i -E "s/^desired_time = */desired_time = $DESIRED_TIME/" maze_cluster_script.py  
    sed -i -E "s/^resume_simulation = (True|False)/resume_simulation = False/" maze_cluster_script.py
    sed -i -E "s/^test_run = (True|False)/test_run = False/" maze_cluster_script.py
 

    echo "=== Running simulation for REAPER_TIMER = $REAPER_TIMER ==="
    rm -f ./data/conc*.txt ./data/part*.txt ./data/*.mp4 ./data/param*.txt ./data/param.txt.bak
    d="./output/reaper_timer/${ER}_emission_rate/${REAPER_TIMER}_until_death"
    mkdir --parents "${d}/data/"
    sed -i -E "s/^[[:space:]]*grim_reaper_delay[[:space:]]*=.*/grim_reaper_delay = $REAPER_TIMER/" maze_cluster_script.py
    python maze_cluster_script.py
    exit_code=$?
    if [ $exit_code -ne 0 ]; then
        sed -i -E "s/^dt = .*/dt = $DT / 2 * 10 ** (-3)  # time step size/" maze_cluster_script.py
        sed -i -E "s/^resume_simulation = False/resume_simulation = True/" maze_cluster_script.py
        python maze_cluster_script.py        
        exit_code=$?
        if [ $exit_code -ne 0 ]; then
            sed -i -E "s/^dt = .*/dt = $DT / 4 * 10 ** (-3)  # time step size/" maze_cluster_script.py
            python maze_cluster_script.py
        fi
        sed -i -E "s/^resume_simulation = True/resume_simulation = False/" maze_cluster_script.py
    fi
    python video_maker_testrun.py
    cp -r ./data/* "${d}/data/"
    cp ./data/param.txt "${d}/${REAPER_TIMER}param.txt"
    cp ./data/param.txt.bak "${d}/${REAPER_TIMER}param.txt.bak"
    mv "${d}/data/particle_trajectory.mp4" "${d}/${REAPER_TIMER}rip_particle_trajectory.mp4"
done

echo "All Reaper Timer simulations complete!"


# ============================================================
# Section 4: exit time statistics
# ============================================================

ER=3.5
DT=0.9
CALCULATED_PARTICLES=$(echo "$ER * 400" | bc | cut -d'.' -f1)
DESIRED_TIME=1000
REAPER_TIMER=12.1
SHOTS=100
STATS_DIR="./output/exit_time_statistics/${ER}_emission_rate/${REAPER_TIMER}"
mkdir -p "$STATS_DIR"

sed -i -E "s/^num_particles = .*/num_particles = $CALCULATED_PARTICLES # Number of particles/" maze_cluster_script.py
sed -i -E "s/^emission_rate = .*/emission_rate = $ER # droplets per second/" maze_cluster_script.py
sed -i -E "s/^dt = .*/dt = $DT * 10 ** (-3) # time step size/" maze_cluster_script.py
sed -i -E "s/^desired_time = .*/desired_time = $DESIRED_TIME/" maze_cluster_script.py
sed -i -E "s/^grim_reaper_delay = .*/grim_reaper_delay = $REAPER_TIMER/" maze_cluster_script.py
sed -i -E "s/^resume_simulation = .*/resume_simulation = False/" maze_cluster_script.py
sed -i -E "s/^test_run = .*/test_run = False/" maze_cluster_script.py

for ((SHOT=1; SHOT<=SHOTS; SHOT++)); do
    echo "=== Shot $SHOT / $SHOTS ==="
    rm -f ./data/conc*.txt ./data/part*.txt ./data/*.mp4 ./data/param*.txt ./data/param.txt.bak

    sed -i -E "s/^dt = .*/dt = $DT * 10 ** (-3) # time step size/" maze_cluster_script.py
    sed -i -E "s/^resume_simulation = .*/resume_simulation = False/" maze_cluster_script.py
    python maze_cluster_script.py
    exit_code=$?

    if [ "$exit_code" -ne 0 ]; then
        sed -i -E "s/^dt = .*/dt = $DT \/ 2 * 10 ** (-3) # time step size/; s/^resume_simulation = .*/resume_simulation = True/" maze_cluster_script.py
        python maze_cluster_script.py
        exit_code=$?
    fi

    if [ "$exit_code" -ne 0 ]; then
        sed -i -E "s/^dt = .*/dt = $DT \/ 4 * 10 ** (-3) # time step size/" maze_cluster_script.py
        python maze_cluster_script.py
        exit_code=$?
    fi

    sed -i -E "s/^dt = .*/dt = $DT * 10 ** (-3) # time step size/; s/^resume_simulation = .*/resume_simulation = False/" maze_cluster_script.py

    [ "$exit_code" -eq 0 ] && [ -f ./data/param.txt.bak ] &&
        python process_exit_statistics.py "$SHOT" "$STATS_DIR"
done

echo "=== Finished $SHOTS statistical shots ==="