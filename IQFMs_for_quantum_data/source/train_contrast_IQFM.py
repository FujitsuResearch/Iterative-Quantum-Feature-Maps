

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
  parser.add_argument('--type_task', type=str, default='classification', help='classification,reg_entropy')

  # For network
  parser.add_argument('--nonlinear', type=str, default='qenc_44rxyzx', help='gelu,relu,qenc_44rxyzx,qenc_44iqp_full')
  parser.add_argument('--rep_enc', type=int, default=1)
  parser.add_argument('--var_depth', type=int, default=2, help='Depth of the pars circuit')
  parser.add_argument('--residual', type=int, default=0, help='Use residual structure or not')
  parser.add_argument('--qenc_in_norm', type=int, default=0, help='Normalize the data before forwarding to the circuit')
  parser.add_argument('--qenc_out_norm', type=int, default=0, help='Normalize the output the circuit')
  
  parser.add_argument('--type_cost', type=str, default='rbf', help='Type of cost function in contrastive learning: diff,rbf')
  parser.add_argument('--type_anchor', type=int, default=1, help='Use anchor or not')
  
  parser.add_argument('--use_BP', type=int, default=0)
  parser.add_argument('--train_qfm', type=int, default=0, help='Training qfm or not')
  parser.add_argument('--layers', type=str, default='784,16,32')
  parser.add_argument('--n_qubits', type=int, default=8, help='Number of wires') 
  parser.add_argument('--noise_level', type=float, default=0.0, help='Noise level added to the data.')

  parser.add_argument('--n_obs', type=int, default=0, help='Number of random observables')
  parser.add_argument('--n_basis', type=int, default=1, help='Number of measurement basis')
  parser.add_argument('--use_record', type=int, default=0, help='Use record of measurement shots (>0) or not (0)')
  parser.add_argument('--n_shots', type=int, default=0, help='0: analytical measurement, >0: specify the number of shots')
  parser.add_argument('--rc', type=int, default=0, help='residual connection')
  

  # For optimizer
  parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
  parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
  parser.add_argument('--scale', type=float, default=8.0, help='Scale for evaluating goodness func')

  # For data
  parser.add_argument('--dat_name', type=str, default='Ising_all', help='Name of the data: Ising_all, Ising_para_para, Ising_para_all, gch, ssh, Ising_all_open')
  parser.add_argument('--n_labels', type=int, default=2, help='Number of labels')
  parser.add_argument('--labels', type=str, default='[0,1]', help='Choose labels')
  parser.add_argument('--p_flip', type=float, default=0.0, help='Probability to flip the labels')
  
  # For training
  parser.add_argument('--n_epochs_outer', type=int, default=200, help='Number of epoch outer')
  parser.add_argument('--n_epochs_inner', type=int, default=40, help='Number of epoch inner')
  
  # For system
  parser.add_argument('--rseed', type=int, default=0, help='Random seed')
  parser.add_argument('--plot_layer', type=int, default=0, help='plot weights of the layer')
  parser.add_argument('--type_samples', type=str, default='rand', help='rand,tensor')

  # For unused
  parser.add_argument('--n_test', type=int, default=10000, help='Number of test data')
  parser.add_argument('--n_train', type=int, default=1000, help='Number of train data')

  args = parser.parse_args()
  
  save_dir, nonlinear, use_BP, n_qubits, n_obs, noise_level = args.save_dir, args.nonlinear, args.use_BP, args.n_qubits, args.n_obs, args.noise_level
  # repeat encoding
  rep_enc, residual, type_cost, type_anchor,  train_qfm = args.rep_enc, args.residual, args.type_cost, args.type_anchor, args.train_qfm
  weight_decay, lr, scale = args.weight_decay, args.lr, args.scale
  rc = args.rc

  dat_name, n_train, n_test, n_labels = args.dat_name, args.n_train, args.n_test, args.n_labels
  n_epochs_outer, n_epochs_inner = args.n_epochs_outer, args.n_epochs_inner
  var_depth, rseed, plot_layer = args.var_depth, args.rseed, args.plot_layer
  use_record, n_shots, n_basis = args.use_record, args.n_shots, args.n_basis
  qenc_in_norm, qenc_out_norm = args.qenc_in_norm, args.qenc_out_norm
  type_samples, p_flip = args.type_samples, args.p_flip

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
  
  basename = f'{dat_name}_{type_samples}_{args.type_task}_{nonlinear_str}_qfm_{train_qfm}_vdepth_{var_depth}_ionorm_{qenc_in_norm}_{qenc_out_norm}_res_{residual}_cost_{type_cost}_anchor_{type_anchor}_layers_{lay_str}_dat_{n_train}_{n_test}_epoch_{n_epochs_outer}_{n_epochs_inner}_scale_{scale}_lr_{lr}_rseed_{rseed}_rc_{rc}_noise_{noise_level}_pflip_{p_flip}'

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

  # Obtain features from all layers
  gather_features = True

  # combined_train_loader_qpm, combined_test_loader_qpm = Dat_QPM(num_qubits)
  # Easy data
  # combined_train_loader_qpm, combined_test_loader_qpm = read_IsingCluster_data(num_qubits, dat_type='all')
  
  # Types of data: ['para', 'anti_ferro', 'all']
  # Combination of using train and test data
  if dat_name == 'Ising_all_open':
      train_dat_type = 'all'
      test_dat_type = 'all'
      combined_train_loader_qpm, combined_test_loader_qpm = read_IsingCluster_data_open(n_qubits, train_dat_type=train_dat_type, test_dat_type=test_dat_type)
  elif "Ising" in dat_name:
    if dat_name == 'Ising_all':
      train_dat_type = 'all'
      test_dat_type = 'all'
    elif dat_name == 'Ising_para_para':
      train_dat_type = 'para'
      test_dat_type = 'para'
    elif dat_name == 'Ising_para_all':
      train_dat_type = 'para'
      test_dat_type = 'all'
    combined_train_loader_qpm, combined_test_loader_qpm = read_IsingCluster_data(n_qubits, train_dat_type=train_dat_type, test_dat_type=test_dat_type)
  elif dat_name == 'gch':
    combined_train_loader_qpm, combined_test_loader_qpm = read_gch_data(n_qubits)
  elif dat_name == 'ssh':
    combined_train_loader_qpm, combined_test_loader_qpm = read_ssh_data(n_qubits)
  elif dat_name == 'dlp_dat_p=251_g=11_n=8':
    dat_file = os.path.join(os.path.dirname(__file__), '../data/dlp', f'{dat_name}.npz')
    p = int(dat_name.split('=')[1].split('_')[0])
    g = int(dat_name.split('=')[2].split('_')[0])
    n_qubits = int(dat_name.split('=')[3])
    sls = [62]
    if not os.path.exists(dat_file):
      raise ValueError(f"Data file {dat_file} does not exist.")
    combined_train_loader_qpm, combined_test_loader_qpm = load_dlp_data(dat_file, p, g, sls, rseed=rseed, batch_size=0)
  elif dat_name == 'dlp_dat_p=251_g=11_n=8_sls':
    dat_file = os.path.join(os.path.dirname(__file__), '../data/dlp', f'{dat_name}.npz')
    p = int(dat_name.split('=')[1].split('_')[0])
    g = int(dat_name.split('=')[2].split('_')[0])
    n_qubits = int(dat_name.split('=')[3].split('_')[0])
    sls = [29, 90, 160]
    if not os.path.exists(dat_file):
      raise ValueError(f"Data file {dat_file} does not exist.")
    combined_train_loader_qpm, combined_test_loader_qpm = load_dlp_data(dat_file, p, g, sls, rseed=rseed, batch_size=0)
  else:
    raise ValueError(f"Unknown dat_name: {dat_name}")

  if noise_level > 0.0:
      # Add unitary noise to training data
      combined_train_loader_qpm = add_X_rot_noise_data_loader(combined_train_loader_qpm, n_wires=n_qubits, noise_level=noise_level)
  if p_flip > 0.0:
      # Flip labels with probability p_flip
      combined_train_loader_qpm = add_label_flip_data_loader(combined_train_loader_qpm, p_flip=p_flip)

  if 'tensor' in type_samples:
    if dat_name == 'gch' or dat_name == 'ssh':
      n_wires = n_qubits + 2
    else:
      n_wires = n_qubits + 1
  else:
    n_wires = n_qubits

  print("n_wires", n_wires)
  print("n_qubits", n_qubits)
  print("Number of training data", len(combined_train_loader_qpm.dataset))
  print("Number of test data", len(combined_test_loader_qpm.dataset))
  
  net_qpm = DeepNet(dims=layers, n_epochs_outer=n_epochs_outer, n_epochs_inner=n_epochs_inner, n_obs=n_obs, noise_level=0.0, use_BP=use_BP,\
          qenc_in_norm=qenc_in_norm, qenc_out_norm=qenc_out_norm, n_basis=n_basis, var_depth=var_depth,
          rep_enc=rep_enc, residual=residual, train_qfm=train_qfm, type_cost=type_cost, type_anchor=type_anchor, n_wires=n_wires, weight_decay=weight_decay, lr=lr,\
          rc=rc, scale=scale, use_record=use_record, n_shots=n_shots, n_labels=n_labels, nonlinear=nonlinear, logger=logger, gather_features=gather_features)
  
  net_qpm.train_qpm(combined_train_loader_qpm, combined_test_loader_qpm, args.type_task, plot_layer=plot_layer, type_samples=type_samples)