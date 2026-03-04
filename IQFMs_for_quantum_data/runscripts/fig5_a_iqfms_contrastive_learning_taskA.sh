export OMP_NUM_THREADS=1

RUN=pyrun_train_DQNN.py
BIN=../source/train_contrast_IQFM.py
SCRIPT_NAME=$(basename $0 .sh)

SEEDS='0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49'
SCALE='8.0'
TASK='classification'
TYPE_SAMPLES='rand'
DAT='Ising_all' # Task A

EPOCH_IN=40
EPOCH_OUT=100 # contrastive_learning
NONLINEAR=pars_qenc_44iqp_full

SHOTS=0
OBS=0
BP=0
NOISE=0.0

NORM='1'  # 0: no norm, 1: sig-norm, 2:tanh, 3:GELU
COST='rbf'
QFM=0 # 1: learn the variational circuit before the feature map, 0: only use the feature map and learn the readout
BASIS=4
RECORD=0
WEIGHT=0.0

LR=0.001

for RC in 1 # 0: re-input OFF, 1: re-input ON
do
for VDEPTH in 2 # depth of the variational circuit
do 
for LAYERS in '32,32' '32,32,32' '32,32,32,32' '32,32,32,32,32' '32,32,32,32,32,32' '32,32,32,32,32,32,32,32,32,32,32'
do
SAVE=../results/$SCRIPT_NAME/$TYPE_SAMPLES\_$TASK\_qfm_$QFM\_basis_$BASIS\_rec_$RECORD\_shots_$SHOTS\_obs_$OBS\_layers_$LAYERS\_lr_$LR\_rc_$RC\_vdepth_$VDEPTH\_epoch_$EPOCH_OUT

python $RUN --bin $BIN --dat_name $DAT --type_task $TASK --type_samples $TYPE_SAMPLES --var_depth $VDEPTH --n_basis $BASIS --noise_level $NOISE --weight_decay $WEIGHT --train_qfm $QFM --type_cost $COST --save_dir $SAVE --qenc_out_norm $NORM --use_BP $BP --n_obs $OBS --use_record $RECORD --n_shots $SHOTS --scale $SCALE --rseed $SEEDS --nonlinear $NONLINEAR --layers $LAYERS --n_epochs_inner $EPOCH_IN --n_epochs_outer $EPOCH_OUT --lr $LR --rc $RC
done 
done
done