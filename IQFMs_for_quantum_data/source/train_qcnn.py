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
from qcnn import *

if __name__ == "__main__":
  # Check for command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--save_dir', type=str, default='../results/del_test_qcnn')

  # For qcnn network
  parser.add_argument('--var_depth', type=int, default=10, help='Depth of the pars circuit')
  
  # For data
  parser.add_argument('--dat_name', type=str, default='Ising_all', help='Name of the data: Ising_all, Ising_para_para, Ising_para_all, gch, ssh')
  parser.add_argument('--n_qubits', type=int, default=8, help='Number of qubits')
  parser.add_argument('--n_labels', type=int, default=2, help='Number of labels')

  parser.add_argument('--noise_level', type=float, default=0.0, help='Noise level added to the data.')
  parser.add_argument('--n_shots', type=int, default=0, help='0: analytical measurement, >0: specify the number of shots')
  
  # For classifier
  parser.add_argument('--out_scale', type=float, default=1.0, help='Scale for output')
  parser.add_argument('--acc_thres', type=float, default=0.5, help='Threshold for accuracy')
  
  # For optimizer
  parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
  parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
  parser.add_argument('--n_epochs', type=int, default=200, help='Number of training epoch')
  parser.add_argument('--batch_rate', type=float, default=1, help='Mini-batch ratio')
  parser.add_argument('--batch_size', type=int, default=0, help='Mini-batch size')

  # For system
  parser.add_argument('--rseed', type=int, default=0, help='Random seed')
  parser.add_argument('--loss_lb', type=str, default='mse', help='Loss label: mse, logis')
  args = parser.parse_args()
  
  save_dir, dat_name, n_qubits, n_labels, noise_level = args.save_dir, args.dat_name, args.n_qubits, args.n_labels, args.noise_level
  weight_decay, lr, n_epochs, batch_rate, batch_size = args.weight_decay, args.lr, args.n_epochs, args.batch_rate, args.batch_size
  
  var_depth, rseed = args.var_depth, args.rseed
  out_scale, acc_thres = args.out_scale, args.acc_thres

  n_shots = args.n_shots

  # Create folder to save results
  log_dir = os.path.join(save_dir,'log' )
  res_dir = os.path.join(save_dir,'res')
  
  os.makedirs(log_dir, exist_ok=True)
  os.makedirs(res_dir, exist_ok=True)
  
  # TRAN (12/27): mse can provide good train and good test?
  loss_lb = args.loss_lb

  # # TRAN (12/27): logis can provide good train but bad test
  # loss_lb = 'logis'
  # acc_thres=0.5

  basename = f'{dat_name}_lbs_{n_labels}_{loss_lb}_vdepth_{var_depth}_epoch_{n_epochs}_lr_{lr}_noise_{noise_level}_rseed_{rseed}_batchrate_{batch_rate}'

  log_filename = os.path.join(log_dir, '{}.log'.format(basename))
  logger = get_module_logger(__name__, log_filename, level='info')
  
  logger.info(log_filename)
  logger.info(args)
  
  torch.set_default_dtype(torch.float32)
  torch.set_printoptions(precision=6)

  random.seed(rseed)
  np.random.seed(rseed)
  torch.manual_seed(rseed)

  # Types of data: ['para', 'anti_ferro', 'all']
  # Combination of using train and test data
  if "Ising" in dat_name:
    if dat_name == 'Ising_all':
      train_dat_type = 'all'
      test_dat_type = 'all'
    elif dat_name == 'Ising_para_para':
      train_dat_type = 'para'
      test_dat_type = 'para'
    elif dat_name == 'Ising_para_all':
      train_dat_type = 'para'
      test_dat_type = 'all'
    combined_train_loader_qpm, combined_test_loader_qpm = read_IsingCluster_data(n_qubits, train_dat_type=train_dat_type, test_dat_type=test_dat_type, batch_size=batch_size)
  elif dat_name == 'gch':
    combined_train_loader_qpm, combined_test_loader_qpm = read_gch_data(n_qubits, batch_size=batch_size)
  elif dat_name == 'ssh':
    combined_train_loader_qpm, combined_test_loader_qpm = read_ssh_data(n_qubits, batch_size=batch_size)

  if noise_level > 0.0:
    # Add unitary noise to training data
    combined_train_loader_qpm = add_X_rot_noise_data_loader(combined_train_loader_qpm, n_wires=n_qubits, noise_level=noise_level)
  
  model = QuantumConvolutionalModel(n_wires=n_qubits, var_depth=var_depth, n_labels=n_labels, n_shots=n_shots)
  for name, param in model.named_parameters():
    logger.info(f"Parameter: {name}, requires_grad: {param.requires_grad}, shape: {param.shape}, dtype: {param.dtype}")
  
  optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

  best_loss = float('inf')  # Track the lowest test loss
  patience = 10  # Number of epochs to wait for improvement
  patience_counter = 0

  for epoch in range(n_epochs):
      model.train()
      # print("len(combined_train_loader_qpm)", len(combined_train_loader_qpm))

      for batch_idx, (batch_data, batch_labels) in enumerate(combined_train_loader_qpm):
          optimizer.zero_grad()

          # Instantiate the quantum device for the batch
          qdev = tq.QuantumDevice(n_wires=n_qubits, bsz=batch_data.size(0), device='cpu')
          # Forward pass
          outputs = model(batch_data, qdev)
          targets = batch_labels.float()
          
          # Compute loss
          loss = batch_loss(targets, outputs, n_labels=n_labels, loss_lb=loss_lb, scale=out_scale)
          
          if model.n_shots == 0:  # Analytical mode for measurement, just backward through the model
              loss.backward()  # Backpropagate

              # # Output gradients
              # for name, param in model.named_parameters():
              #     if param.grad is not None:
              #         print(f"Gradient for {name}: {param.grad}")
              #         print(f"paramdata for {name}: {param.data}")
          else:      
              # 1) compute classical-only backprop for FC
              if model.fc is not None:
                # first, we need to freeze the quantum encoder parameters
                for p in model.encoder.params:
                    p.requires_grad = False
                loss.backward()  # Only backpropagate the loss through the FC layers
              
                # Unfreeze the quantum encoder parameters
                for p in model.encoder.params:
                    p.requires_grad = True
              
              # 2) get QCNN encoder grads by parameter shift
              qc_grads  = parameter_shift(qdev, model, batch_data, batch_labels, loss_lb, out_scale)

              for param, grad in zip(model.encoder.params, qc_grads):
                grad = grad.to(param.device)
                param.grad = grad

          optimizer.step()  # Update parameters

      if epoch == 0 or epoch % 10 == 9:
      # if epoch == 0 or (epoch + 1) % 200 == 0:
          # Compute accuracy
          train_accuracy, train_loss = compute_accuracy(model, combined_train_loader_qpm, scale=out_scale, loss_lb=loss_lb, acc_thres=acc_thres)
          test_accuracy, test_loss   = compute_accuracy(model, combined_test_loader_qpm, scale=out_scale, loss_lb=loss_lb, acc_thres=acc_thres)
          logger.info(f"Epoch {epoch+1}: Train loss={train_loss:.6f}; Test loss={test_loss:.6f}; Train accuracy={train_accuracy:.2f}% ; Test accuracy={test_accuracy:.2f}%")

          if test_loss < best_loss:
              best_loss = test_loss
              patience_counter = 0
          else:
              patience_counter += 1
          
          if patience_counter >= patience:
              logger.info(f"Early stopping triggered at epoch {epoch + 1}. Best test loss={best_loss}, accuracy={test_accuracy:.2f}%")
              break