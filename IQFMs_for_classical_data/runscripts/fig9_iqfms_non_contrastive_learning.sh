export OMP_NUM_THREADS=1

RUN=pyrun_train_DQNN.py
BIN=../source/train_contrast_IQFM.py
SCRIPT_NAME=$(basename $0 .sh)

SEEDS='0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49'
SCALE='8.0'
NTEST=10000


EPOCH_IN=40
EPOCH_OUT=0
NONLINEAR=qenc_44iqp_full

SHOTS=0
OBS=0
BP=0
NORM='1'  #0: no norm, 1: sig-norm, 2:tanh, 3:GELU
COST='rbf'
QFM=0
BASIS=4
RECORD=0
WEIGHT=0.0
LABELS='[0,1,2,3,4,5,6,7,8,9]' # choose labels of Fashion-MNIST
N_LABELS=10
ANCHOR='no_use' 

for NTRAIN in 5000 
do
for LAYERS in '784,16,16,16,16,16' '784,64,64,64,64,64' '784,256,256,256,256,256'
do
SAVE=../results/$SCRIPT_NAME/fmnist_qenc_$COST\_qfm_$QFM\_basis_$BASIS\_record_$RECORD\_shots_$SHOTS\_obs_$OBS\_labels_$LABELS\_n_train_$NTRAIN\_n_test_$NTEST\_layers_$LAYERS\_type_anchor_$ANCHOR

python $RUN --bin $BIN --n_basis $BASIS --weight_decay $WEIGHT --train_qfm $QFM --type_cost $COST --save_dir $SAVE --qenc_out_norm $NORM --use_BP $BP --n_obs $OBS --use_record $RECORD --n_shots $SHOTS --scale $SCALE --rseed $SEEDS --n_train $NTRAIN --n_test $NTEST --nonlinear $NONLINEAR --layers $LAYERS --n_epochs_inner $EPOCH_IN --n_epochs_outer $EPOCH_OUT --labels $LABELS --n_labels $N_LABELS --type_anchor $ANCHOR
done
done