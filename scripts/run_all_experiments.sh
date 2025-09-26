#!/bin/bash

# Run All Experiments Script
# This script runs each experiment configuration and captures output to experiment_logs/

set -eE  # Exit on any error and trap errors
trap 'echo "Error occurred at line $LINENO with exit code $?"' ERR

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create experiment_logs directory if it doesn't exist
mkdir -p experiment_logs

# Get the absolute path to the experiment_logs directory
LOG_DIR="$(pwd)/experiment_logs"

# Parse command line arguments
FORCE_MODE=false
MAX_JOBS=2  # Default to 2 parallel jobs
while [[ $# -gt 0 ]]; do
    case $1 in
        --force|-f)
            FORCE_MODE=true
            shift
            ;;
        --jobs|-j)
            if [[ -n "$2" && "$2" =~ ^[0-9]+$ ]]; then
                MAX_JOBS="$2"
                shift 2
            else
                echo -e "${RED}Error: --jobs requires a numeric argument${NC}"
                exit 1
            fi
            ;;
        *)
            break
            ;;
    esac
done

echo -e "${BLUE}=== Lightning-Hydra-Template-Extended Experiment Runner ===${NC}"
echo -e "${BLUE}Log directory: ${LOG_DIR}${NC}"
if [ "$FORCE_MODE" = true ]; then
    echo -e "${YELLOW}Force mode: ON (ignoring all experiment markers)${NC}"
fi
echo -e "${BLUE}Parallel jobs: ${MAX_JOBS}${NC}"
echo ""

# Array of all experiment names (without .yaml extension)
# Template experiments - ready to run fresh
experiments=(
    "example"
    "trivial_micro_small"
    "trivial_micro_small_regression"
    "trivial_tiny_small"
    "trivial_vit_micro_small"
    "wah_cnn_medium_regression"
    "wah_cnn_medium"
    "wah_cnn_tiny_ordinal"
    "wah_cnn_tiny_quantized"
    "wah_cnn_tiny_regression"
    "wah_cnn_tiny_soft_target"
    "wah_cnn_tiny_weighted"
    "wah_cnn_tiny"
    "wah_cnn_tiny_auxiliary"
    "wah_cnn_tiny_auxiliary_regression"
    "wah_cnn_medium_auxiliary"
    "wah_cnn_medium_auxiliary_regression"
    "wah_vit_medium_regression"
    "wah_vit_medium"
    "wah_vit_tiny_regression"
    "wah_vit_tiny"
)

# Function to parse experiment name and return (marker, name)
parse_experiment() {
    local experiment_entry="$1"

    # Check if experiment has a marker (Y/N prefix)
    if [[ "$experiment_entry" =~ ^([YN])\ (.+)$ ]]; then
        local marker="${BASH_REMATCH[1]}"
        local name="${BASH_REMATCH[2]}"
        echo "$marker $name"
    else
        echo "$experiment_entry"
    fi
}

# Function to run experiments in parallel with job control
run_parallel_experiments() {
    experiment_list=("$@")
    local active_jobs=0
    local job_pids=()
    local job_names=()
    local job_modes=()

    for experiment in "${experiment_list[@]}"; do
        # Parse the experiment entry
        local parsed=$(parse_experiment "$experiment")
        local marker=$(echo "$parsed" | cut -d' ' -f1)
        if [[ "$parsed" =~ ^[YN]\  ]]; then
            # Has marker, extract name from field 2
            local experiment_name=$(echo "$parsed" | cut -d' ' -f2-)
        else
            # No marker, use the full parsed string
            local experiment_name="$parsed"
        fi

        # Determine mode based on processing mode and marker
        local mode="run"
        if [ "$processing_mode" = "force" ]; then
            mode="run"
        else
            case "$marker" in
                "Y") mode="skip" ;;
                "N") mode="debug" ;;
                *) mode="run" ;;  # Default for unmarked experiments
            esac
        fi

        # Run experiment in background
        run_experiment "$experiment_name" "$mode" &
        local job_pid=$!
        job_pids+=($job_pid)
        job_names+=("$experiment")
        job_modes+=("$mode")

        active_jobs=$((active_jobs + 1))

        # Wait if we've reached max jobs
        if [ $active_jobs -ge $MAX_JOBS ]; then
            local job_finished=false
            while [ "$job_finished" = false ]; do
                for i in "${!job_pids[@]}"; do
                    local pid="${job_pids[i]}"
                    if [ -z "$pid" ]; then
                        continue
                    fi
                    if ! kill -0 "$pid" 2>/dev/null; then
                        local exit_code
                        if wait "$pid"; then
                            exit_code=0
                        else
                            exit_code=$?
                        fi

                        local job_mode="${job_modes[i]}"
                        if [ "$job_mode" = "skip" ]; then
                            skipped=$((skipped + 1))
                        elif [ $exit_code -eq 0 ]; then
                            completed=$((completed + 1))
                        else
                            failed=$((failed + 1))
                        fi

                        unset job_pids[i]
                        unset job_names[i]
                        unset job_modes[i]
                        active_jobs=$((active_jobs - 1))
                        job_finished=true
                        break
                    fi
                done

                if [ "$job_finished" = false ]; then
                    sleep 0.2
                fi
            done
        fi
    done

    # Wait for all remaining jobs to complete
    while [ $active_jobs -gt 0 ]; do
        local job_finished=false
        for i in "${!job_pids[@]}"; do
            local pid="${job_pids[i]}"
            if [ -z "$pid" ]; then
                continue
            fi
            if ! kill -0 "$pid" 2>/dev/null; then
                local exit_code
                if wait "$pid"; then
                    exit_code=0
                else
                    exit_code=$?
                fi

                local job_mode="${job_modes[i]}"
                if [ "$job_mode" = "skip" ]; then
                    skipped=$((skipped + 1))
                elif [ $exit_code -eq 0 ]; then
                    completed=$((completed + 1))
                else
                    failed=$((failed + 1))
                fi

                unset job_pids[i]
                unset job_names[i]
                unset job_modes[i]
                active_jobs=$((active_jobs - 1))
                job_finished=true
                break
            fi
        done

        if [ "$job_finished" = false ]; then
            sleep 0.2
        fi
    done
}

# Function to extract and display errors from log file
show_errors_from_log() {
    local log_file="$1"
    local experiment_name="$2"

    if [ ! -f "$log_file" ]; then
        return
    fi

    # Extract error lines from the log file (lines containing common error patterns)
    local error_lines=$(grep -E "(Error|ERROR|Exception|EXCEPTION|Traceback|Failed|FAILED|fatal|FATAL)" "$log_file" | head -20)

    if [ -n "$error_lines" ]; then
        echo -e "${RED}[$(date '+%H:%M:%S')] Error details for ${experiment_name}:${NC}"
        echo -e "${YELLOW}--- Last 20 error lines from log ---${NC}"
        echo "$error_lines"
        echo -e "${YELLOW}--- End error extract ---${NC}"
        echo -e "${BLUE}Full log: ${log_file}${NC}"
        echo ""
    fi
}

# Function to run a single experiment
run_experiment() {
    local experiment_name="$1"
    local mode="${2:-run}"  # Default mode is "run" if not specified
    local log_file="${LOG_DIR}/${experiment_name}-log.txt"

    case "$mode" in
        "skip")
            echo -e "${BLUE}[$(date '+%H:%M:%S')] Skipping experiment: ${experiment_name} (marked as successful)${NC}"
            return 0
            ;;
        "debug")
            echo -e "${YELLOW}[$(date '+%H:%M:%S')] Debugging experiment: ${experiment_name} (marked as failed)${NC}"
            echo -e "${BLUE}  Output will be saved to: ${log_file}${NC}"
            echo -e "${YELLOW}  Debug mode: Will show detailed error information${NC}"
            ;;
        "run")
            echo -e "${YELLOW}[$(date '+%H:%M:%S')] Starting experiment: ${experiment_name}${NC}"
            echo -e "${BLUE}  Output will be saved to: ${log_file}${NC}"
            ;;
        *)
            echo -e "${RED}Error: Invalid mode '$mode' for experiment '$experiment_name'${NC}"
            return 1
            ;;
    esac

    # Check if log file already exists and rename it if so
    if [ -f "${log_file}" ]; then
        # Try Linux stat format first, then macOS format
        local mod_time=""
        if command -v stat >/dev/null 2>&1; then
            # Linux format
            mod_time=$(stat -c %Y "${log_file}" 2>/dev/null)
            if [ $? -eq 0 ] && [ -n "${mod_time}" ]; then
                # Convert epoch time to YYYY-MM-DD-HH-MM-SS format
                local formatted_date=$(date -d "@${mod_time}" +%Y-%m-%d-%H-%M-%S 2>/dev/null)
                if [ $? -eq 0 ] && [ -n "${formatted_date}" ]; then
                    local backup_file="${log_file%.*}-${formatted_date}.txt"
                    echo -e "${YELLOW}  Existing log file found, renaming to: ${backup_file}${NC}"
                    mv "${log_file}" "${backup_file}"
                else
                    # Fallback to current timestamp
                    local backup_file="${log_file%.*}-$(date +%Y-%m-%d-%H-%M-%S).txt"
                    echo -e "${YELLOW}  Existing log file found, renaming to: ${backup_file}${NC}"
                    mv "${log_file}" "${backup_file}"
                fi
            else
                # Try macOS format
                mod_time=$(stat -f %Sm -t %Y%m%d%H%M%S "${log_file}" 2>/dev/null)
                if [ $? -eq 0 ] && [ -n "${mod_time}" ]; then
                    # Convert YYYYMMDDHHMMSS to YYYY-MM-DD-HH-MM-SS format
                    local formatted_date=$(echo "${mod_time}" | sed -E 's/([0-9]{4})([0-9]{2})([0-9]{2})([0-9]{2})([0-9]{2})([0-9]{2})/\1-\2-\3-\4-\5-\6/')
                    local backup_file="${log_file%.*}-${formatted_date}.txt"
                    echo -e "${YELLOW}  Existing log file found, renaming to: ${backup_file}${NC}"
                    mv "${log_file}" "${backup_file}"
                else
                    # Final fallback: use current timestamp
                    local backup_file="${log_file%.*}-$(date +%Y-%m-%d-%H-%M-%S).txt"
                    echo -e "${YELLOW}  Existing log file found, renaming to: ${backup_file}${NC}"
                    mv "${log_file}" "${backup_file}"
                fi
            fi
        else
            # Final fallback: use current timestamp
            local backup_file="${log_file%.*}-$(date +%Y-%m-%d-%H-%M-%S).txt"
            echo -e "${YELLOW}  Existing log file found, renaming to: ${backup_file}${NC}"
            mv "${log_file}" "${backup_file}"
        fi
    fi

    # Add experiment header to log file
    {
        echo "==================================================================="
        echo "EXPERIMENT: ${experiment_name}"
        echo "STARTED: $(date)"
        echo "COMMAND: time python src/train.py experiment=${experiment_name}"
        echo "==================================================================="
        echo ""
    } > "${log_file}"

    # Run the experiment and capture both stdout and stderr
    if [ "$mode" = "debug" ]; then
        echo -e "${YELLOW}  Executing: time python src/train.py experiment=${experiment_name}${NC}"
        echo -e "${YELLOW}  Debug mode: Detailed output follows...${NC}"
        echo ""

        # In debug mode, show real-time output and capture to log (including time output)
        { time python src/train.py experiment="${experiment_name}"; } 2>&1 | tee -a "${log_file}"
        local exit_code=${PIPESTATUS[0]}

        echo ""
        echo -e "${YELLOW}  Debug mode: Execution completed with exit code $exit_code${NC}"

        # If debug mode failed, rename the log file to include _FAILED
        if [ $exit_code -ne 0 ]; then
            local log_file_without_extension="${log_file%.*}"
            local log_file_extension="${log_file##*.}"
            local failed_log_file="${log_file_without_extension}_FAILED.${log_file_extension}"
            mv "${log_file}" "${failed_log_file}"
            echo -e "${YELLOW}  Debug mode failed - log moved to: ${failed_log_file}${NC}"
            echo -e "${YELLOW}  (Errors were already displayed above in debug mode)${NC}"
        fi
    else
        # Normal mode - capture output to log file (including time output)
        { time python src/train.py experiment="${experiment_name}"; } >> "${log_file}" 2>&1
        local exit_code=$?
    fi

    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}[$(date '+%H:%M:%S')] ✓ Completed: ${experiment_name}${NC}"
        # Add completion footer to log file
        {
            echo ""
            echo "==================================================================="
            echo "EXPERIMENT COMPLETED SUCCESSFULLY"
            echo "FINISHED: $(date)"
            echo "==================================================================="
        } >> "${log_file}"
    else
        echo -e "${RED}[$(date '+%H:%M:%S')] ✗ Failed: ${experiment_name} (exit code: $exit_code)${NC}"

        # Create failed log file name (insert _FAILED before .txt)
        local log_file_without_extension="${log_file%.*}"
        local log_file_extension="${log_file##*.}"
        local failed_log_file="${log_file_without_extension}_FAILED.${log_file_extension}"

        # Move log file to failed log file first
        mv "${log_file}" "${failed_log_file}"
        echo -e "${YELLOW}  Failed log saved to: ${failed_log_file}${NC}"

        # Add failure footer to failed log file
        {
            echo ""
            echo "==================================================================="
            echo "EXPERIMENT FAILED"
            echo "EXIT CODE: $exit_code"
            echo "FINISHED: $(date)"
            echo "==================================================================="
        } >> "${failed_log_file}"

        # Show errors from the failed log file
        show_errors_from_log "${failed_log_file}" "${experiment_name}"
    fi

    echo ""
}

# Function to run experiments with optional filtering
run_experiments() {
    local start_time=$(date +%s)
    local total_experiments=${#experiments[@]}
    # Make these global so they can be accessed from run_parallel_experiments
    completed=0
    failed=0
    skipped=0

    echo -e "${BLUE}Total experiments to process: ${total_experiments}${NC}"
    echo ""

    # Determine processing mode for all experiments (global for run_parallel_experiments)
    processing_mode=""
    if [ "$FORCE_MODE" = true ]; then
        processing_mode="force"
        echo -e "${YELLOW}Force mode: All experiments will be run normally${NC}"
    else
        processing_mode="normal"
    fi
    echo ""

    # If arguments provided, run only specified experiments
    if [ $# -gt 0 ]; then
        echo -e "${YELLOW}Running specified experiments: $@${NC}"

        # Filter experiments to only those specified
        local filtered_experiments=()
        for experiment_spec in "$@"; do
            if [[ " ${experiments[@]} " =~ " ${experiment_spec} " ]]; then
                filtered_experiments+=("$experiment_spec")
            else
                echo -e "${RED}Warning: Experiment '${experiment_spec}' not found in experiment list${NC}"
            fi
        done

        # Run filtered experiments in parallel
        run_parallel_experiments "${filtered_experiments[@]}"
    else
        # Run all experiments
        if [ "$processing_mode" = "force" ]; then
            echo -e "${YELLOW}Running all experiments in force mode...${NC}"
        else
            echo -e "${YELLOW}Processing all experiments...${NC}"
        fi

        # Run all experiments in parallel
        run_parallel_experiments "${experiments[@]}"
    fi

    # Summary
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local hours=$((duration / 3600))
    local minutes=$(((duration % 3600) / 60))
    local seconds=$((duration % 60))

    echo -e "${BLUE}=== EXPERIMENT SUMMARY ===${NC}"
    echo -e "${GREEN}Completed: ${completed}${NC}"
    echo -e "${RED}Failed: ${failed}${NC}"
    echo -e "${BLUE}Skipped: ${skipped}${NC}"
    echo -e "${BLUE}Total processed: $((completed + failed + skipped))${NC}"
    echo -e "${BLUE}Total time: ${hours}h ${minutes}m ${seconds}s${NC}"
    echo -e "${BLUE}Logs saved in: ${LOG_DIR}${NC}"
}

# Help function
show_help() {
    echo "Usage: $0 [experiment_names...]"
    echo ""
    echo "Run Neural-Spectral-Modeling-Template experiments and save logs."
    echo ""
    echo "Optional Experiment Marking Scheme (in the experiment-list in the source code):"
    echo "  Y experiment_name    - Skip (already successful)"
    echo "  N experiment_name    - Debug mode (failed, needs attention)"
    echo "  experiment_name      - Run normally (no marking)"
    echo ""
    echo "Options:"
    echo "  No arguments         - Process all experiments according to markings"
    echo "  experiment_names     - Process only specified experiments"
    echo "  --force, -f         - Ignore markers and run all experiments normally"
    echo "  --jobs, -j NUM      - Run NUM experiments in parallel (default: 2)"
    echo "  --list              - List all available experiments"
    echo "  --help              - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                # Run all experiments not marked 'Y'"
    echo "  $0 wah_cnn example                # Process specific experiments"
    echo "  $0 \"N wah_cnn_tiny_regression\"  # Debug a failed experiment (marked 'N')"
    echo "  $0 --force                        # Run all experiments ignoring markers"
    echo "  $0 -f wah_cnn                     # Force-run specific experiment"
    echo "  $0 --jobs 4                       # Run 4 experiments in parallel"
    echo "  $0 -j 8 --force                   # Run 8 experiments in parallel, ignore markers"
    echo "  $0 --list                         # List available experiments"
    echo ""
    echo "Log files are saved to: experiment_logs/EXPERIMENT_NAME-log.txt"
    echo "Debug mode shows real-time output and captures detailed error information."
}

# List experiments function
list_experiments() {
    echo -e "${BLUE}Available experiments:${NC}"
    for experiment in "${experiments[@]}"; do
        # Parse the experiment entry
        local parsed=$(parse_experiment "$experiment")
        local marker=$(echo "$parsed" | cut -d' ' -f1)
        if [[ "$parsed" =~ ^[YN]\  ]]; then
            # Has marker, extract name from field 2
            local name=$(echo "$parsed" | cut -d' ' -f2-)
        else
            # No marker, use the full parsed string
            local name="$parsed"
        fi

        case "$marker" in
            "Y")
                echo -e "  ${GREEN}✓${NC} ${experiment} ${BLUE}(successful)${NC}"
                ;;
            "N")
                echo -e "  ${RED}✗${NC} ${experiment} ${YELLOW}(failed)${NC}"
                ;;
            " ")
                echo -e "  ${BLUE}•${NC} ${experiment} ${BLUE}(pending)${NC}"
                ;;
        esac
    done
    echo ""
    echo -e "${BLUE}Total: ${#experiments[@]} experiments${NC}"
    echo ""
    echo -e "${BLUE}Legend:${NC}"
    echo -e "  ${GREEN}✓${NC} = Successful (will be skipped)"
    echo -e "  ${RED}✗${NC} = Failed (will be debugged)"
    echo -e "  ${BLUE}•${NC} = Pending (will be run normally)"
}

# Parse command line arguments
case "${1:-}" in
    --help|-h)
        show_help
        exit 0
        ;;
    --list|-l)
        list_experiments
        exit 0
        ;;
    *)
        run_experiments "$@"
        ;;
esac
