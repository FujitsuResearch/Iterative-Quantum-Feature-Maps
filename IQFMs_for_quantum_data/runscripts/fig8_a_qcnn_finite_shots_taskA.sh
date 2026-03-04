#!/bin/bash

export OMP_NUM_THREADS=1

# Script and binary paths
RUN=pyrun_train_QCNN.py
BIN=../source/train_qcnn.py
SCRIPT_NAME=$(basename $0 .sh)

# Parameters
SEEDS='0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49'

DAT='Ising_all'  # Dataset name TaskA
N_LABELS=2 # Number of label for TaskA
EPOCH_OUT=300  # Number of training epochs
NOISE=0.0  # Noise level
WEIGHT=0.0  # Weight decay
N_QUBITS=8  # Number of qubits
SCALE=1.0  # Output scale



for THRES in 0.4  # Threshold for classification
do
for VDEPTH in 4  # Depth of the variational circuit
do
for SHOTS in 1000 500 100 10
do
for BATCH_SIZE in 4
do
for LR in 0.001
do
    # Save directory for results
    SAVE=../results/$SCRIPT_NAME/$DAT\_vdepth_$VDEPTH\_lr_$LR\_epoch_$EPOCH_OUT\_noise_$NOISE\_scale_$SCALE\_thres_$THRES\_shots_$SHOTS\_batchsize_$BATCH_SIZE
    
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
        --acc_thres $THRES \
        --n_shots $SHOTS \
        --batch_size $BATCH_SIZE \
        --n_labels $N_LABELS
done
done
done
done
done