#!/bin/bash

# =================================================================
#  Script for Comparative Experiments under Label Noise & Class Imbalance
#  Compares CE vs LS / Baseline vs Fixed ETF Classifier
# =================================================================

# Activate Python virtual environment (Modify the path if necessary)
# source venv/bin/activate

# --- Experiment Settings ---
IMBALANCE_RATIO=0.1     # Imbalance ratio (1.0 = balanced, e.g., 0.1 for 1:10 ratio between minority/majority)
NOISE_RATIO=0.1         # Label noise ratio (0.0 = clean, e.g., 0.1 for 10% noise)
LS_EPS=0.05             # Label smoothing epsilon value (only used when loss='ls')
BASE_EXP_NAME="resnet18" # Prefix for experiment names

echo "===== Starting Experiments: Imbalance Ratio ${IMBALANCE_RATIO}, Noise Ratio ${NOISE_RATIO} ====="

# Loop through all combinations
for imbalanced in "no" "yes"; do # Imbalanced or Balanced
  for noisy in "no" "yes"; do     # Noisy or Clean labels
    for fixed_fc in "no" "yes"; do # Learnable or Fixed ETF FC
      for loss_type in "ce" "ls"; do # Cross-Entropy or Label Smoothing

        # --- Set Parameters based on loop variables ---
        current_imbalance_ratio=${IMBALANCE_RATIO}
        imbalance_tag="imbal${IMBALANCE_RATIO}"
        if [ "$imbalanced" = "no" ]; then
          current_imbalance_ratio=1.0 # Use balanced dataset
          imbalance_tag="bal"         # Tag for balanced
        fi

        current_noise_ratio=${NOISE_RATIO}
        noise_tag="noise${NOISE_RATIO}"
        if [ "$noisy" = "no" ]; then
          current_noise_ratio=0.0 # Use clean labels
          noise_tag="clean"       # Tag for clean
        fi

        etf_flag=""
        fc_type="learnFC"
        if [ "$fixed_fc" = "yes" ]; then
          etf_flag="--ETF_fc"
          fc_type="fixedETF"
        fi

        loss_flag="--loss ${loss_type}"
        eps_flag=""
        ls_tag=""
        if [ "$loss_type" = "ls" ]; then
            eps_flag="--eps ${LS_EPS}"
            ls_tag="_ls${LS_EPS}"
        fi


        # --- Generate Experiment Name ---
        # Example: resnet18_imbal0.1_noise0.1_fixedETF_ls0.05
        EXP_NAME="${BASE_EXP_NAME}_${imbalance_tag}_${noise_tag}_${fc_type}${ls_tag}"

        echo ""
        echo "--- Running Experiment: ${EXP_NAME} ---"
        echo "Config: Imbalanced=$imbalanced(${current_imbalance_ratio}), Noisy=$noisy(${current_noise_ratio}), FixedFC=$fixed_fc, Loss=$loss_type"

        # --- Execute main.py ---
        python main.py \
            --dset cifar10 \
            --model resnet18 \
            ${loss_flag} \
            ${eps_flag} \
            --wd 5e-4 \
            --scheduler ms \
            --max_epochs 800 \
            --batch_size 128 \
            --lr 0.05 \
            --log_freq 10 \
            --exp_name "${EXP_NAME}" \
            --imbalance_ratio ${current_imbalance_ratio} \
            --noise_ratio ${current_noise_ratio} \
            ${etf_flag} \
            --save_ckpt 100 # Save checkpoints every 100 epochs

      done # End loss_type loop
    done # End fixed_fc loop
  done # End noisy loop
done # End imbalanced loop

echo ""
echo "===== All Experiments Completed ====="