

import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
from collections import defaultdict
import random

import os
import argparse
import io
import sys
import itertools

import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

from utils import *
from model_base import Model
from effective_dimension import EffectiveDimension
from IQFM_model import DeepNet

if __name__ == "__main__":
  # Check for command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--save_dir', type=str, default='../results/test_iqfm_eff')

  # For network
  parser.add_argument('--nonlinear', type=str, default='sigmoid', help='sigmoid,tanh,gelu,relu,qenc_44rxyzx')
  parser.add_argument('--rep_enc', type=int, default=1)
  parser.add_argument('--residual', type=int, default=0, help='Use residual structure or not')
  parser.add_argument('--qenc_in_norm', type=int, default=0, help='Normalize the data before forwarding to the circuit')
  parser.add_argument('--qenc_out_norm', type=int, default=0, help='Normalize the output the circuit')
  
  parser.add_argument('--type_cost', type=str, default='rbf', help='Type of cost function in contrastive learning: diff,rbf')
  
  parser.add_argument('--use_BP', type=int, default=0)
  parser.add_argument('--train_qfm', type=int, default=0, help='Training qfm or not')
  parser.add_argument('--layers', type=str, default='16,16,16')
  parser.add_argument('--n_wires', type=int, default=4, help='Number of wires')
  parser.add_argument('--n_obs', type=int, default=0, help='Number of random observables')
  parser.add_argument('--n_basis', type=int, default=1, help='Number of measurement basis')
  parser.add_argument('--use_record', type=int, default=0, help='Use record of measurement shots (>0) or not (0)')
  parser.add_argument('--n_shots', type=int, default=0, help='0: analytical measurement, >0: specify the number of shots')
  

  # For optimizer
  parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
  parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
  parser.add_argument('--scale', type=float, default=8.0, help='Scale for evaluating goodness func')

  # For data
  parser.add_argument('--dat_name', type=str, default='Uni', help='Name of the data')
  parser.add_argument('--n_inputs', type=int, default=100, help='Number of input data')
  parser.add_argument('--n_thetas', type=int, default=10, help='Number of parameter sets')
  parser.add_argument('--n_labels', type=int, default=10, help='Number of labels')
  
  # For training
  parser.add_argument('--n_epochs_outer', type=int, default=200, help='Number of epoch outer')
  parser.add_argument('--n_epochs_inner', type=int, default=40, help='Number of epoch inner')
  
  # For system
  parser.add_argument('--rseed', type=int, default=0, help='Random seed')
  
  args = parser.parse_args()
  
  save_dir, nonlinear, use_BP, n_wires, n_obs = args.save_dir, args.nonlinear, args.use_BP, args.n_wires, args.n_obs
  # repeat encoding
  rep_enc, residual, type_cost, train_qfm = args.rep_enc, args.residual, args.type_cost, args.train_qfm
  weight_decay, lr, scale = args.weight_decay, args.lr, args.scale
  dat_name, n_inputs, n_thetas, n_labels = args.dat_name, args.n_inputs, args.n_thetas, args.n_labels
  n_epochs_outer, n_epochs_inner = args.n_epochs_outer, args.n_epochs_inner
  rseed = args.rseed
  use_record, n_shots, n_basis = args.use_record, args.n_shots, args.n_basis
  qenc_in_norm, qenc_out_norm = args.qenc_in_norm, args.qenc_out_norm

  layers = [int(x) for x in args.layers.split(',')]
  lay_str = '_'.join([str(x) for x in layers])

  # Create folder to save results
  res_dir = os.path.join(save_dir,'res')
  os.makedirs(res_dir, exist_ok=True)

  if 'qenc' in nonlinear:
    if n_basis > 1:
      nonlinear_str = f'{nonlinear}_basis_{n_basis}'
    else:
      nonlinear_str = f'{nonlinear}'
    nonlinear_str = f'{nonlinear_str}_BP_{use_BP}_obs_{n_obs}_shots_{n_shots}_rec_{use_record}'
    if rep_enc > 1:
      nonlinear_str = f'{nonlinear_str}_rep_{rep_enc}'
  else:
    nonlinear_str = f'{nonlinear}'
  
  basename = f'{nonlinear_str}_qfm_{train_qfm}_ionorm_{qenc_in_norm}_{qenc_out_norm}_res_{residual}_layers_{lay_str}_dat_{n_inputs}_nthetas_{n_thetas}_rseed_{rseed}'

  torch.set_default_dtype(torch.double)
  random.seed(rseed)
  np.random.seed(rseed)
  torch.manual_seed(rseed)

  if dat_name == 'FMnist':
    train_loader, val_loader = Dat_FMnist(n_inputs, n_inputs, n_inputs, n_inputs, rseed=rseed)
    dat_input = []
    for features, _ in train_loader:
      dat_input.append(features)
    dat_input = torch.cat(dat_input, dim=0)
  else:
    # use random data
    dat_input = None
  net = DeepNet(dims=layers, n_epochs_outer=n_epochs_outer, n_epochs_inner=n_epochs_inner, n_obs=n_obs, use_BP=use_BP,\
          qenc_in_norm=qenc_in_norm, qenc_out_norm=qenc_out_norm, n_basis=n_basis,
          rep_enc=rep_enc, residual=residual, train_qfm=train_qfm, type_cost=type_cost, n_wires=n_wires, weight_decay=weight_decay, lr=lr,\
          scale=scale, use_record=use_record, n_shots=n_shots, n_labels=n_labels, nonlinear=nonlinear)
  
  ed = EffectiveDimension(net, num_thetas=n_thetas, num_inputs=n_inputs, seed=rseed, x = dat_input)
  f, trace = ed.get_fhat()
  # range of the number of data
  ndat_ls = [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000]
  effdim = ed.eff_dim(f, ndat_ls)
  norm_effdim = np.array(effdim) / ed.d
  #print('Effdim', effdim)
  #print('Normalized effdim', norm_effdim)
  res_file = os.path.join(res_dir, f'{basename}_eff.txt')
  np.savetxt(res_file, norm_effdim)