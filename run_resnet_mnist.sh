#!/bin/bash

# =================================================================
#  Hypothesis Verification Experiments (Parallel)
#  - Target Conditions:
#    1. Clean + CE (Fixed vs Learnable)
#    2. Noisy + CE (Fixed vs Learnable)
#    3. Noisy + LS (Fixed vs Learnable)
# =================================================================

# Activate Python virtual environment
# source venv/bin/activate

# --- Target Settings (Change these as needed) ---
TARGET_MODEL="resnet18"
TARGET_LS_EPS=0.05

declare -a TARGET_DATASETS=("mnist")
declare -a SEEDS=(2021)

# --- Parallel Execution Settings ---
MAX_PARALLEL_JOBS=1     # Max concurrent jobs
CPU_THREADS_PER_JOB=2   # Threads per job (MKL/OMP/OPENBLAS)

# --- Experiment Parameters ---
# Imbalance ratios to test (e.g., 0.1, 0.01)
declare -a IMBALANCE_RATIOS=(0.1 0.01)
# Noise ratios to test (0.0=Clean, >0.0=Noisy)
declare -a NOISE_RATIOS=(0.0 0.1)
# Fixed FC options
declare -a FIXED_FC_OPTIONS=("yes" "no")
# Loss types
declare -a LOSS_TYPES=("ce" "ls")

# --- Other Fixed Settings ---
MAX_EPOCHS=800
BATCH_SIZE=128
LEARNING_RATE=0.01
WEIGHT_DECAY=5e-4
LOG_FREQ=1
SAVE_CKPT=100

echo "===== Starting Hypothesis Verification (Parallel) ====="
echo "Model: ${TARGET_MODEL}, Datasets: ${TARGET_DATASETS[*]}, LS Eps: ${TARGET_LS_EPS}, Seeds: ${SEEDS[*]}"
echo "Max Jobs: ${MAX_PARALLEL_JOBS}, Threads/Job: ${CPU_THREADS_PER_JOB}"

# Loop through all combinations
for current_seed in "${SEEDS[@]}"; do
  for current_dataset in "${TARGET_DATASETS[@]}"; do
    for current_imbalance_ratio in "${IMBALANCE_RATIOS[@]}"; do
      imbalance_tag="imbal${current_imbalance_ratio}"
      if awk -v ratio="$current_imbalance_ratio" 'BEGIN { exit !(ratio == 1.0) }'; then
        imbalance_tag="bal"
      fi

      for current_noise_ratio in "${NOISE_RATIOS[@]}"; do
        noise_tag="noise${current_noise_ratio}"
        is_clean=0
        if awk -v ratio="$current_noise_ratio" 'BEGIN { exit !(ratio == 0.0) }'; then
          noise_tag="clean"
          is_clean=1
        fi

        for fixed_fc in "${FIXED_FC_OPTIONS[@]}"; do
          etf_flag=""
          fc_type="learnFC"
          if [ "$fixed_fc" = "yes" ]; then
            etf_flag="--ETF_fc"
            fc_type="fixedETF"
          fi

          for loss_type in "${LOSS_TYPES[@]}"; do
            # --- Filter Conditions ---
            # Skip "LS" loss if data is "Clean" (We only want to test LS recovery on Noisy data)
            if [ "$loss_type" = "ls" ] && [ "$is_clean" -eq 1 ]; then
                continue
            fi
            # -------------------------

            loss_flag="--loss ${loss_type}"
            eps_flag=""
            ls_tag=""
            if [ "$loss_type" = "ls" ]; then
              eps_flag="--eps ${TARGET_LS_EPS}"
              ls_tag="_ls${TARGET_LS_EPS}"
            elif [ "$loss_type" = "dr" ]; then
              ls_tag="_dr"
            fi

            # Experiment Name
            EXP_NAME="${TARGET_MODEL}_${imbalance_tag}_${noise_tag}_${fc_type}${ls_tag}"

            # --- Parallel Execution Logic ---
            # Check job count and wait
            if [[ $(jobs -r -p | wc -l) -ge $MAX_PARALLEL_JOBS ]]; then
                echo "--- Waiting for a slot (Running: $(jobs -r -p | wc -l)) ---"
                wait -n
            fi

            echo ""
            echo "--- Launching: ${EXP_NAME} ---"
            echo "Config: Imbal=${current_imbalance_ratio}, Noise=${current_noise_ratio}, FC=${fc_type}, Loss=${loss_type}"

            # Run in background subshell with environment variables set
            (
              export OPENBLAS_NUM_THREADS=${CPU_THREADS_PER_JOB}
              export MKL_NUM_THREADS=${CPU_THREADS_PER_JOB}
              export OMP_NUM_THREADS=${CPU_THREADS_PER_JOB}
              
              python main.py \
                  --dset ${current_dataset} \
                  --model ${TARGET_MODEL} \
                  --seed ${current_seed} \
                  ${loss_flag} \
                  ${eps_flag} \
                  --wd ${WEIGHT_DECAY} \
                  --scheduler ms \
                  --max_epochs ${MAX_EPOCHS} \
                  --batch_size ${BATCH_SIZE} \
                  --lr ${LEARNING_RATE} \
                  --log_freq ${LOG_FREQ} \
                  --exp_name "${EXP_NAME}" \
                  --imbalance_ratio ${current_imbalance_ratio} \
                  --noise_ratio ${current_noise_ratio} \
                  ${etf_flag} \
                  --save_ckpt ${SAVE_CKPT}
            ) & 

          done # loss loop
        done # fc loop
      done # noise loop
    done # imbalance loop  
  done # seed loop
done # dataset loop

echo ""
echo "--- All jobs launched. Waiting for completion... ---"
wait
echo "===== All Experiments Completed ====="