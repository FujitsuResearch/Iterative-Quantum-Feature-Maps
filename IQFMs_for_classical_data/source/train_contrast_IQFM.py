

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
from loginit import get_module_logger
from IQFM_model import DeepNet

if __name__ == "__main__":
  # Check for command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--save_dir', type=str, default='../results/del_test_iqfm_rbf')

  # For network
  parser.add_argument('--nonlinear', type=str, default='qenc_44rxyzx', help='gelu,relu,qenc_44rxyzx,qenc_44iqp_full')
  parser.add_argument('--rep_enc', type=int, default=1)
  parser.add_argument('--var_depth', type=int, default=2, help='Depth of the pars circuit')
  parser.add_argument('--residual', type=int, default=0, help='Use residual structure or not')
  parser.add_argument('--qenc_in_norm', type=int, default=0, help='Normalize the data before forwarding to the circuit')
  parser.add_argument('--qenc_out_norm', type=int, default=0, help='Normalize the output the circuit')
  
  parser.add_argument('--type_cost', type=str, default='rbf', help='Type of cost function in contrastive learning: diff,rbf')
  parser.add_argument('--type_anchor', type=str, default='rand', help='Type of anchor in contrastive learning: rand, rotation_all_data, rotation_one_data_per_label, rotation_one_data_per_data')
  
  parser.add_argument('--use_BP', type=int, default=0)
  parser.add_argument('--train_qfm', type=int, default=0, help='Training qfm or not')
  parser.add_argument('--layers', type=str, default='784,16,32')
  parser.add_argument('--n_wires', type=int, default=4, help='Number of wires')
  parser.add_argument('--noise_level', type=float, default=0.0, help='Noise level added to the data.')
  parser.add_argument('--n_obs', type=int, default=0, help='Number of random observables')
  parser.add_argument('--n_basis', type=int, default=1, help='Number of measurement basis')
  parser.add_argument('--use_record', type=int, default=0, help='Use record of measurement shots (>0) or not (0)')
  parser.add_argument('--n_shots', type=int, default=0, help='0: analytical measurement, >0: specify the number of shots')
  

  # For optimizer
  parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
  parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
  parser.add_argument('--scale', type=float, default=8.0, help='Scale for evaluating goodness func')

  # For data
  parser.add_argument('--dat_name', type=str, default='Fashion-Mnist', help='Name of the data')
  parser.add_argument('--n_test', type=int, default=10000, help='Number of test data')
  parser.add_argument('--n_train', type=int, default=1000, help='Number of train data')
  parser.add_argument('--n_labels', type=int, default=10, help='Number of labels')
  parser.add_argument('--labels', type=str, default='[0,1,2,3,4,5,6,7,8,9]', help='Choose labels')
  
  # For training
  parser.add_argument('--n_epochs_outer', type=int, default=200, help='Number of epoch outer')
  parser.add_argument('--n_epochs_inner', type=int, default=40, help='Number of epoch inner')
  
  # For system
  parser.add_argument('--rseed', type=int, default=0, help='Random seed')
  
  args = parser.parse_args()
  
  save_dir, nonlinear, use_BP, n_wires, n_obs, noise_level = args.save_dir, args.nonlinear, args.use_BP, args.n_wires, args.n_obs, args.noise_level
  # repeat encoding
  rep_enc, residual, type_cost, type_anchor,  train_qfm = args.rep_enc, args.residual, args.type_cost, args.type_anchor, args.train_qfm
  weight_decay, lr, scale = args.weight_decay, args.lr, args.scale
  dat_name, n_train, n_test, n_labels = args.dat_name, args.n_train, args.n_test, args.n_labels
  n_epochs_outer, n_epochs_inner = args.n_epochs_outer, args.n_epochs_inner
  var_depth, rseed = args.var_depth, args.rseed
  use_record, n_shots, n_basis = args.use_record, args.n_shots, args.n_basis
  qenc_in_norm, qenc_out_norm = args.qenc_in_norm, args.qenc_out_norm

  layers = [int(x) for x in args.layers.split(',')]
  lay_str = '_'.join([str(x) for x in layers])
  labels = eval(args.labels)

  # Create folder to save results
  log_dir = os.path.join(save_dir,'log' )
  res_dir = os.path.join(save_dir,'res')
  
  os.makedirs(log_dir, exist_ok=True)
  os.makedirs(res_dir, exist_ok=True)
  
  if 'qenc' in nonlinear:
    if n_basis > 1:
      nonlinear_str = f'{nonlinear}_basis_{n_basis}'
    else:
      nonlinear_str = f'{nonlinear}'
    nonlinear_str = f'{nonlinear_str}_BP_{use_BP}_obs_{n_obs}_shots_{n_shots}'
  else:
    nonlinear_str = f'{nonlinear}'
  
  basename = f'{nonlinear_str}_qfm_{train_qfm}_ionorm_{qenc_in_norm}_{qenc_out_norm}_res_{residual}_cost_{type_cost}_anchor_{type_anchor}_layers_{lay_str}_dat_{n_train}_{n_test}_epoch_{n_epochs_outer}_{n_epochs_inner}_scale_{scale}_lr_{lr}_rseed_{rseed}_noise_level_{noise_level}_vdepth_{var_depth}'

  log_filename = os.path.join(log_dir, '{}.log'.format(basename))
  logger = get_module_logger(__name__, log_filename, level='info')
  
  logger.info(log_filename)
  logger.info(args)
  
  torch.set_default_dtype(torch.float32)
  
  #train_loader, val_loader = Dat_FMnist(10000, 10000)
  #net = DeepNet([28*28, 1000, 1000, 1000], n_epochs_outer=300, n_epochs_inner=25, n_labels=10)

  random.seed(rseed)
  np.random.seed(rseed)
  torch.manual_seed(rseed)

  # combine the original images and the 90-degree rotated images to shuffle them so that the order corresponds
  combined_train_loader, combined_val_loader = Dat_FMnist(n_train, n_test, n_train, n_test, rseed=rseed, labels=labels, noise_level=noise_level)
  net = DeepNet(dims=layers, n_epochs_outer=n_epochs_outer, n_epochs_inner=n_epochs_inner, n_obs=n_obs, use_BP=use_BP,\
          qenc_in_norm=qenc_in_norm, qenc_out_norm=qenc_out_norm, n_basis=n_basis, var_depth=var_depth,
          rep_enc=rep_enc, residual=residual, train_qfm=train_qfm, type_cost=type_cost, type_anchor=type_anchor, n_wires=n_wires, weight_decay=weight_decay, lr=lr,\
          scale=scale, use_record=use_record, n_shots=n_shots, n_labels=n_labels, nonlinear=nonlinear, logger=logger)
  
  net.train(combined_train_loader, combined_val_loader)