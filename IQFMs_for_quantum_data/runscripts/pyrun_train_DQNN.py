#!/usr/bin/env python
# # -*- coding: utf-8 -*-

import numpy as np 
import os 
import subprocess
import multiprocessing
import argparse

# Framework params
class FParams():
    def __init__(self, n_epochs_outer, n_epochs_inner):
        self.n_epochs_outer = n_epochs_outer
        self.n_epochs_inner = n_epochs_inner

# Network params
class NParams():
    def __init__(self, nonlinear, use_BP, layers, n_qubits, use_record, n_shots, n_obs, n_basis,\
            residual, qenc_in_norm, qenc_out_norm, train_qfm, type_cost, type_anchor, rc, noise_level, var_depth):
        self.nonlinear = nonlinear
        self.use_BP = use_BP
        self.layers = layers
        self.n_qubits = n_qubits
        self.use_record = use_record
        self.n_shots = n_shots
        self.n_obs = n_obs
        self.n_basis = n_basis
        self.residual = residual
        self.qenc_in_norm = qenc_in_norm
        self.qenc_out_norm = qenc_out_norm
        self.train_qfm = train_qfm
        self.type_cost = type_cost
        self.type_anchor = type_anchor
        self.rc = rc
        self.noise_level = noise_level
        self.var_depth = var_depth

# Data params
class DParams():
    def __init__(self, dat_name, n_train, n_test, n_labels, labels, p_flip):
        self.dat_name = dat_name
        self.n_train = n_train
        self.n_test = n_test
        self.n_labels = n_labels
        self.labels = labels
        self.p_flip = p_flip

# Optimizer params
class OParams():
    def __init__(self, weight_decay, lr, scale):        
        self.weight_decay = weight_decay
        self.lr = lr
        self.scale = scale

def execute_job(bin, type_task, fparams, nparams, dparams, oparams, save_dir, plot_layer, type_samples, rseed):
    print(f'Start process with rseed={rseed}')
    cmd = f'python {bin} \
            --n_epochs_outer {fparams.n_epochs_outer} \
            --n_epochs_inner {fparams.n_epochs_inner} \
            --nonlinear {nparams.nonlinear} \
            --use_BP {nparams.use_BP} \
            --layers {nparams.layers} \
            --n_qubits {nparams.n_qubits} \
            --use_record {nparams.use_record} \
            --n_shots {nparams.n_shots} \
            --n_obs {nparams.n_obs} \
            --n_basis {nparams.n_basis} \
            --residual {nparams.residual} \
            --type_cost {nparams.type_cost} \
            --qenc_in_norm {nparams.qenc_in_norm} \
            --qenc_out_norm {nparams.qenc_out_norm} \
            --train_qfm {nparams.train_qfm} \
            --type_cost {nparams.type_cost} \
            --type_anchor {nparams.type_anchor} \
            --rc {nparams.rc} \
            --noise_level {nparams.noise_level} \
            --var_depth {nparams.var_depth} \
            --weight_decay {oparams.weight_decay} \
            --lr {oparams.lr} \
            --scale {oparams.scale} \
            --dat_name {dparams.dat_name} \
            --n_test {dparams.n_test} \
            --n_train {dparams.n_train} \
            --n_labels {dparams.n_labels} \
            --labels {dparams.labels} \
            --p_flip {dparams.p_flip} \
            --rseed {rseed} \
            --save_dir {save_dir} \
            --type_task {type_task} \
            --plot_layer {plot_layer} \
            --type_samples {type_samples}'
    os.system(cmd)
    print(f'Finish process with rseed={rseed}')
    
if __name__ == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--bin', type=str, default='../source/DQNN.py')
    parser.add_argument('--save_dir', type=str, default='../results/fmnist_1226')
    parser.add_argument('--type_task', type=str, default='classification', help='classification,reg_entropy')

    # For network
    parser.add_argument('--nonlinear', type=str, default='gelu', help='gelu, relu, qenc1')
    parser.add_argument('--use_BP', type=int, default=0)
    parser.add_argument('--residual', type=int, default=0, help='Use residual structure or not')
    parser.add_argument('--qenc_in_norm', type=int, default=0, help='Normalize the data before forwarding to the circuit')
    parser.add_argument('--qenc_out_norm', type=str, default='1', help='Normalize the output the circuit')
    parser.add_argument('--var_depth', type=int, default=2, help='Depth of the pars circuit')

    parser.add_argument('--type_cost', type=str, default='diff', help='Type of cost function in contrastive learning: diff,rbf')
    parser.add_argument('--type_anchor', type=int, default=1, help='Use anchor or not')
    parser.add_argument('--train_qfm', type=int, default=1, help='Training qfm or not')

    parser.add_argument('--layers', type=str, default='784,16,16,16')
    parser.add_argument('--n_qubits', type=int, default=8, help='Number of wires') # デフォルトを8に変更
    parser.add_argument('--use_record', type=int, default=0, help='Use record of measurement shots (>0) or not (0)')
    parser.add_argument('--n_shots', type=int, default=0, help='0: analytical measurement, >0: specify the number of shots')
    parser.add_argument('--n_obs', type=int, default=16, help='Number of random observables')
    parser.add_argument('--n_basis', type=int, default=1, help='Number of measurement basis')
    parser.add_argument('--rc', type=int, default=0, help='residual connection')
    parser.add_argument('--noise_level', type=float, default=0.0, help='Noise level added to the data.')

    # For optimizer
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--scale', type=str, default='8.0', help='Scale for evaluating goodness func')

    # For data
    parser.add_argument('--dat_name', type=str, default='Ising_all', help='Name of the data: Ising_all, Ising_para')
    parser.add_argument('--n_labels', type=int, default=2, help='Number of labels')
    parser.add_argument('--labels', type=str, default='[0,1]', help='Choose labels')
    parser.add_argument('--p_flip', type=float, default=0.0, help='Probability to flip the labels')

    # For training
    parser.add_argument('--n_epochs_outer', type=int, default=200, help='Number of epoch outer')
    parser.add_argument('--n_epochs_inner', type=int, default=40, help='Number of epoch inner')
    
    # For system
    parser.add_argument('--rseed', type=str, default='0', help='Random seed')
    parser.add_argument('--plot_layer', type=int, default=0, help='plot weights of the layer')
    parser.add_argument('--type_samples', type=str, default='rand', help='rand,tensor')

    # NOT USED
    parser.add_argument('--n_test', type=int, default=10000, help='Number of test data')
    parser.add_argument('--n_train', type=int, default=1000, help='Number of train data')

    args = parser.parse_args()

    save_dir, use_BP, n_qubits = args.save_dir, args.use_BP, args.n_qubits
    weight_decay, lr, scale = args.weight_decay, args.lr, args.scale
    rc, noise_level, var_depth, p_flip = args.rc, args.noise_level, args.var_depth, args.p_flip
    dat_name, n_train, n_test, n_labels, labels = args.dat_name, args.n_train, args.n_test, args.n_labels, args.labels
    n_epochs_outer, n_epochs_inner = args.n_epochs_outer, args.n_epochs_inner
    train_qfm, type_cost, type_anchor = args.train_qfm, args.type_cost, args.type_anchor
    type_task, plot_layer, type_samples = args.type_task, args.plot_layer, args.type_samples

    rseeds = [int(x) for x in args.rseed.split(',')]
    scales = [float(x) for x in args.scale.split(',')]
    nonlinear_ls = [str(x) for x in args.nonlinear.split(',')]
    qenc_out_norm_ls = [int(x) for x in args.qenc_out_norm.split(',')]
    layers = args.layers

    fparams = FParams(n_epochs_outer, n_epochs_inner)
    dparams = DParams(dat_name=dat_name, n_train=n_train, n_test=n_test, n_labels=n_labels, labels=labels, p_flip=p_flip)
    
    jobs = []
    for qenc_out_norm in qenc_out_norm_ls:
        for nonlinear in nonlinear_ls:
            nparams = NParams(nonlinear=nonlinear, use_BP=use_BP, layers=layers, n_qubits=n_qubits, n_basis=args.n_basis, var_depth=var_depth,\
                use_record=args.use_record, n_shots=args.n_shots, n_obs=args.n_obs, train_qfm=train_qfm, type_cost=type_cost,\
                type_anchor=type_anchor, residual=args.residual, qenc_in_norm=args.qenc_in_norm, qenc_out_norm=qenc_out_norm, rc=rc, noise_level=noise_level)
            for scale in scales:
                oparams = OParams(weight_decay=weight_decay, lr=lr, scale=scale)
                for rseed in rseeds:
                    p = multiprocessing.Process(target=execute_job, args=(args.bin, type_task, fparams, nparams, dparams, oparams, save_dir, plot_layer, type_samples, rseed))
                    jobs.append(p)
        
    # Start the process
    for p in jobs:
        p.start()

    # Ensure all processes have finished execution
    for p in jobs:
        p.join()