#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import multiprocessing
import argparse

# Framework params
class FParams:
    def __init__(self, n_epochs_outer, n_epochs_inner):
        self.n_epochs_outer = n_epochs_outer
        self.n_epochs_inner = n_epochs_inner

# Network params
class NParams:
    def __init__(self, var_depth, n_qubits, noise_level, n_shots):
        self.var_depth = var_depth
        self.n_qubits = n_qubits
        self.noise_level = noise_level
        self.n_shots = n_shots

# Data params
class DParams:
    def __init__(self, dat_name, n_labels):
        self.dat_name = dat_name
        self.n_labels = n_labels

# Optimizer params
class OParams:
    def __init__(self, weight_decay, lr, out_scale, acc_thres, batch_rate, batch_size):
        self.weight_decay = weight_decay
        self.lr = lr
        self.out_scale = out_scale
        self.acc_thres = acc_thres
        self.batch_rate = batch_rate
        self.batch_size = batch_size

def execute_job(bin, fparams, nparams, dparams, oparams, save_dir, rseed):
    """
    Executes a single job by calling the train_qcnn script with the given parameters.
    """
    print(f"Starting process with rseed={rseed}")
    cmd = f'python3 {bin} \
            --save_dir {save_dir} \
            --var_depth {nparams.var_depth} \
            --n_qubits {nparams.n_qubits} \
            --noise_level {nparams.noise_level} \
            --dat_name {dparams.dat_name} \
            --weight_decay {oparams.weight_decay} \
            --lr {oparams.lr} \
            --out_scale {oparams.out_scale} \
            --acc_thres {oparams.acc_thres} \
            --n_epochs {fparams.n_epochs_outer} \
            --rseed {rseed} \
            --n_shots {nparams.n_shots} \
            --n_labels {dparams.n_labels} \
            --batch_rate {oparams.batch_rate} \
            --batch_size {oparams.batch_size}'
            
    os.system(cmd)
    print(f"Finished process with rseed={rseed}")

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--bin', type=str, default='../source/train_qcnn.py', help='Path to the train_qcnn script')
    parser.add_argument('--save_dir', type=str, default='../results/qcnn_results', help='Directory to save results')
    parser.add_argument('--var_depth', type=int, default=10, help='Depth of the variational circuit')
    parser.add_argument('--n_qubits', type=int, default=8, help='Number of qubits')
    parser.add_argument('--noise_level', type=float, default=0.0, help='Noise level in the quantum data')
    parser.add_argument('--dat_name', type=str, default='Ising_all', help='Dataset name (e.g., Ising_all, Ising_para_all)')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay for the optimizer')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--n_epochs_outer', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--rseed', type=str, default='0', help='Random seeds (comma-separated)')
    parser.add_argument('--n_labels', type=int, default=2, help='Number of labels')
    parser.add_argument('--n_shots', type=int, default=0, help='0: analytical measurement, >0: specify the number of shots')
    parser.add_argument('--batch_rate', type=float, default=1, help='Mini-batch ratio')
    parser.add_argument('--batch_size', type=int, default=0, help='Mini-batch size')


    # For classifier
    parser.add_argument('--out_scale', type=float, default=1.0, help='Scale for output')
    parser.add_argument('--acc_thres', type=float, default=0.5, help='Threshold for accuracy')


    args = parser.parse_args()

    # Parse random seeds
    rseeds = [int(x) for x in args.rseed.split(',')]

    # Prepare parameter objects
    fparams = FParams(n_epochs_outer=args.n_epochs_outer, n_epochs_inner=None)  # n_epochs_inner is unused here
    nparams = NParams(var_depth=args.var_depth, n_qubits=args.n_qubits, noise_level=args.noise_level, n_shots=args.n_shots)
    dparams = DParams(dat_name=args.dat_name, n_labels=args.n_labels)
    oparams = OParams(weight_decay=args.weight_decay, lr=args.lr, out_scale=args.out_scale, acc_thres=args.acc_thres, batch_rate=args.batch_rate, batch_size=args.batch_size)

    # Create and run jobs
    jobs = []
    for rseed in rseeds:
        p = multiprocessing.Process(
            target=execute_job,
            args=(args.bin, fparams, nparams, dparams, oparams, args.save_dir, rseed)
        )
        jobs.append(p)

    # Start all processes
    for p in jobs:
        p.start()

    # Wait for all processes to complete
    for p in jobs:
        p.join()
