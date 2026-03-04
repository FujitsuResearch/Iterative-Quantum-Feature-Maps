: <<'COMMENT'
When using UMAP, please restore the following commented-out code in Iterative-Quantum-Feature-Maps/IQFMs_for_quantum_data/source/IQFM_model.py.
# import umap  

# umap_model = umap.UMAP(
#     n_components=2,
#     random_state=0
# )
# umap_result = umap_model.fit_transform(feats)


# # Save the results to a log file.
# if self.logger is not None:
#   self.logger.info(f'umap_result:{umap_result.tolist()}')   
#   self.logger.info(f'input_dim:{input_dim}') 
#   self.logger.info(f'train_labels_qpm:{labels}')
COMMENT


export OMP_NUM_THREADS=1

RUN=pyrun_train_DQNN.py
BIN=../source/train_contrast_IQFM.py
SCRIPT_NAME=$(basename $0 .sh)

SEEDS='0'
SCALE='8.0'
TASK='classification'
TYPE_SAMPLES='rand'
DAT='Ising_all' # Task A
N_LABELS=2
N_QUBITS=8

EPOCH_IN=40
EPOCH_OUT=100
NONLINEAR=pars_qenc_44iqp_full


SHOTS=0
OBS=0
BP=0
NOISE=0.0

NORM='1'  #0: no norm, 1: sig-norm, 2:tanh, 3:GELU
COST='rbf'
QFM=0 #1: learn the variational circuit before the feature map, 0: only use the feature map and learn the readout
BASIS=4
RECORD=0
WEIGHT=0.0

LR=0.001

for RC in 1 # 0: re-input OFF, 1: re-input ON
do
for VDEPTH in 2 #depth of the variational circuit
do 
for LAYERS in '32,32,32,32,32,32' #layers of the feature map
do
SAVE=../results/$SCRIPT_NAME/$TYPE_SAMPLES\_$TASK\_qfm_$QFM\_basis_$BASIS\_rec_$RECORD\_shots_$SHOTS\_obs_$OBS\_layers_$LAYERS\_lr_$LR\_rc_$RC\_vdepth_$VDEPTH\_noise_$NOISE\_epoch_$EPOCH_OUT\_n_labels_$N_LABELS\_n_qubits_$N_QUBITS

python $RUN --bin $BIN --dat_name $DAT --type_task $TASK --type_samples $TYPE_SAMPLES --var_depth $VDEPTH --n_basis $BASIS --noise_level $NOISE --weight_decay $WEIGHT --train_qfm $QFM --type_cost $COST --save_dir $SAVE --qenc_out_norm $NORM --use_BP $BP --n_obs $OBS --use_record $RECORD --n_shots $SHOTS --scale $SCALE --rseed $SEEDS --nonlinear $NONLINEAR --layers $LAYERS --n_epochs_inner $EPOCH_IN --n_epochs_outer $EPOCH_OUT --lr $LR --rc $RC --n_labels $N_LABELS --n_qubits $N_QUBITS
done
done
done