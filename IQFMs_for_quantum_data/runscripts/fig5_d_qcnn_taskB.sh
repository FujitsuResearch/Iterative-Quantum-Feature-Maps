#!/bin/bash

export OMP_NUM_THREADS=1

# Script and binary paths
RUN=pyrun_train_QCNN.py
BIN=../source/train_qcnn.py
SCRIPT_NAME=$(basename $0 .sh)

# Parameters
SEEDS='0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49'

DAT='gch'  # Dataset name Task B
N_LABELS=4
EPOCH_OUT=5000  # Number of training epochs

WEIGHT=0.0  # Weight decay
LR=0.001  # Learning rate
N_QUBITS=8  # Number of qubits
SCALE=1.0  # Output scale

for THRES in 0.5  # Threshold for classification
do
for VDEPTH in 1 2 4 8 16 32  # Depth of the variational circuit
do
for NOISE in 0.00 # Noise level
do
    # Save directory for results
    SAVE=../results/$SCRIPT_NAME/$DAT\_vdepth_$VDEPTH\_lr_$LR\_epoch_$EPOCH_OUT\_noise_$NOISE\_scale_$SCALE\_n_labels_$N_LABELS
    
    # Run the training script
    python $RUN \
        --bin $BIN \
        --save_dir $SAVE \
        --dat_name $DAT \
        --var_depth $VDEPTH \
        --n_qubits $N_QUBITS \
        --noise_level $NOISE \
        --weight_decay $WEIGHT \
        --lr $LR \
        --n_epochs_outer $EPOCH_OUT \
        --rseed $SEEDS \
        --out_scale $SCALE \
        --n_labels $N_LABELS
done
done
done
