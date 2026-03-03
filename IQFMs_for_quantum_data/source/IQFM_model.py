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
from iqfm_utils import *
from model_base import Model
from effective_dimension import EffectiveDimension
from loginit import get_logger_parent_folder

# For torch quantum
import torchquantum as tq
import torchquantum.functional as tqf
from torchquantum.measurement import expval_joint_analytical, expval_joint_sampling_grouping

from torch.utils.data import TensorDataset, DataLoader

# from sklearn.decomposition import PCA
# import umap

class Layer(nn.Linear):
    def __init__(self, linear_in_fes, linear_out_fes, 
                 weight_decay, lr, rc, scale, n_epochs_inner, n_wires, n_shots, use_record, n_basis,
                 noise_level, n_obs, use_BP, rep_enc, residual, type_cost, type_anchor, train_qfm, qenc_in_norm, qenc_out_norm, force_set_lin_in=False, 
                 bias=True, device=None, dtype=None, use_cuda=False, nonlinear=GELU, var_depth=2):
        super().__init__(linear_in_fes, linear_out_fes, bias, device, dtype)

        self.linear_in_fes = linear_in_fes
        self.scale = scale
        self.use_cuda = use_cuda
        self.nonlinear = nonlinear
        self.n_wires = n_wires
        self.n_shots = n_shots
        self.n_basis = n_basis

        # If use_record = 1, use measurement record instead of expectation value
        self.use_record = use_record
        self.noise_level = noise_level
        self.n_obs = n_obs
        self.use_BP = use_BP
        self.rep_enc = rep_enc
        self.residual = residual
        self.type_cost = type_cost
        self.type_anchor = type_anchor
        self.train_qfm = train_qfm
        self.qenc_in_norm = qenc_in_norm
        self.qenc_out_norm = qenc_out_norm

        self.n_epochs_inner = n_epochs_inner
        self.train_loss = 0.0
        self.weight_decay = weight_decay
        self.lr = lr
        self.rc = rc
        # Depth of variational layer
        self.var_depth = var_depth

        # Random encode layers
        self.rand_enc_params = np.random.uniform(0, 2 * np.pi, size=2 * self.n_wires * self.var_depth)  # Generate a random angle.

        # Parameter for variational layer
        #self.var_params = nn.Parameter(torch.rand(4 * self.n_wires * self.var_depth) * 2 * np.pi)
        self.var_params = nn.ParameterList([
            nn.Parameter(torch.rand(1) * 2 * np.pi) for _ in range(4 * self.n_wires * self.var_depth)
        ])
        
        # thetamin: minimum used in uniform sampling of the parameters
        # thetamax: maximum used in uniform sampling of the parameters
        self.thetamin = 0
        self.thetamax = 1

        if self.train_qfm > 0:
          num_nonlinear_out = linear_out_fes
        else:
          num_nonlinear_out = linear_in_fes

        # For activation function
        if self.nonlinear == RELU:
          self.act_func = torch.nn.ReLU()
        elif self.nonlinear == GELU:
          self.act_func = torch.nn.GELU()
        elif self.nonlinear == SGM:
          self.act_func = torch.nn.Sigmoid()
        elif self.nonlinear == TANH:
          self.act_func = torch.nn.Tanh()
        elif self.nonlinear == ID:
          self.act_func = torch.nn.Identity()
        elif 'qenc' in self.nonlinear:
          # Measure for quantum circuit
          self.measure = tq.MeasureAll(tq.PauliZ)
          # Designed gates
          if 'qenc_44iqp_full' in self.nonlinear:
            self.encoder_gates = [tqf.rz] * self.n_wires + [tqf.rzz] * self.n_wires + [tqf.rz] * self.n_wires + [tqf.rzz] * self.n_wires + \
             [tqf.rz] * self.n_wires + [tqf.rzz] * self.n_wires + [tqf.rz] * self.n_wires + [tqf.rzz] * self.n_wires
            # self.encoder_gates = [tqf.rx] * self.n_wires + [tqf.ry] * self.n_wires + \
            #               [tqf.rz] * self.n_wires + [tqf.rx] * self.n_wires
            self.n_fet_enc = self.n_wires * 4
            self.Hadamard = tq.Hadamard(has_params=False, trainable=False)
          else:
            # Generate encoder,   n_fet_enc  is the umber of features encoded in each circuit
            self.encoder, self.n_fet_enc = gen_encoder(self.n_wires, self.nonlinear)
            
            # self.rx0 = tq.RX(has_params=True, trainable=True)
            # self.ry0 = tq.RY(has_params=True, trainable=True)
            # self.rz0 = tq.RZ(has_params=True, trainable=True)
            # self.rzz0 = tq.RZZ(has_params=True, trainable=True)

          n_c = int(linear_out_fes / self.n_fet_enc)
          # if rands_ in the encoding circuit, add random layers to increase the expressivity
          if 'rands_' in self.nonlinear:
            self.random_layers = []
            for _ in range(n_c):
              self.random_layers.append(tq.RandomLayer(
                    n_ops=self.n_wires**2, wires=list(range(self.n_wires))
              ))

          # if the number of basis > 1, we perform random rotation on each qubit first before measuring in Z-basis
          if self.n_basis > 1:
            self.random_rotXs = []
            for _ in range(1, self.n_basis):
              tqx = tq.RX(has_params=True, trainable=False)
              # reset to the random params
              tqx.reset_params()
              #print('Reset', tqx.params)
              self.random_rotXs.append(tqx)
              
          # Select the measurement operators
          obs_all = []
          if self.use_record == 1 and self.n_shots > 0:
            # Use classical shadow, more reference from https://pennylane.ai/qml/demos/tutorial_classical_shadows/
            #1. The quantum state \rho s prepared with a circuit
            #2. A unitary U is randomly selected from the ensemble and applied to \rho
            #3. A computational basis measurement is performed
            #4. The snapshot is recorded as the observed eigenvalue 1, -1 for |0>, |1>, respectively
            # and the index of the randomly selected

            # prepare the complete set of available Pauli operators
            # applying the single-qubit Clifford circuit is equivalent to measuring a Pauli
            unitary_ops = ["X", "Y", "Z"]
            # sample random Pauli measurements uniformly, where 0,1,2 = X,Y,Z
            # note that the information unitary_ensmb is important also (since it is the index of unitary)
            unitary_ensmb = np.random.randint(0, 3, size=(self.n_shots, n_wires), dtype=int)

            # we create the obs_all to obtain the mesurement record
            for ns in range(self.n_shots):
              obs_ls = []
              for i in range(n_wires):
                obs = []
                for j in range(n_wires):
                  if j == i:
                    obs.append(unitary_ops[unitary_ensmb[ns, i]])
                  else:
                    obs.append("I")
                obs_ls.append("".join(obs))
              obs_all.append(obs_ls)
          elif self.use_record == 2:
            # using Z basis measurement but with correlated Z to 
            # increase the number of features
            prod_ls = list(itertools.product('IZ', repeat = n_wires))
            for obs in prod_ls[1:]:
              obs_all.append("".join(obs))
          else:
            # create a list of measurement operators, each measuring a single qubit in the Z basis.
            for i in range(n_wires):
                obs = ['I'] * n_wires
                obs[i] = 'Z'
                obs_all.append("".join(obs))

          self.obs_all = obs_all
          # print(f'qenc in self.nonlinear Pauli obs: {obs_all}')

          num_nonlinear_out = get_num_out_features(linear_out_fes, self.nonlinear, self.n_wires, \
            self.n_obs, self.n_shots, self.use_record, self.n_fet_enc, self.n_basis)
          n_c = int(linear_out_fes / self.n_fet_enc)
          self.n_fet_out = int(num_nonlinear_out/n_c)
          
          # self.Q = nn.Linear(num_nonlinear_out, linear_out_fes, bias=True)
          # if use_cuda:
          #  self.Q = self.Q.cuda()
          # print(f'n_fet_enc={self.n_fet_enc}, n_fet_out={self.n_fet_out}, linear_in_fes={linear_in_fes}, linear_out_fes={linear_out_fes}, num_nonlinear_out={num_nonlinear_out}')
        elif 'quantum_data' in self.nonlinear:
          # print("self.nonlinear == quantum_data:encoder_gates")
          # Measure for quantum circuit
          self.measure = tq.MeasureAll(tq.PauliZ)
          n_wires_qpm = self.n_wires

          self.noise_gates = [tqf.rx] * n_wires_qpm
          self.encoder_gates = [tqf.rx] * n_wires_qpm + [tqf.rzz] * n_wires_qpm
          self.Hadamard = tq.Hadamard(has_params=False, trainable=False)

          # if the number of basis > 1, we perform random rotation on each qubit first before measuring in Z-basis 
          if self.n_basis > 1:
            self.random_rotXs = []
            for _ in range(1, self.n_basis):
              tqx = tq.RX(has_params=True, trainable=False)
              # reset to the random params
              tqx.reset_params()
              #print('Reset', tqx.params)
              self.random_rotXs.append(tqx)
              
          # Select the measurement operators
          obs_all = []
          if self.use_record == 1 and self.n_shots > 0:
            # Use classical shadow, more reference from https://pennylane.ai/qml/demos/tutorial_classical_shadows/
            #1. The quantum state \rho s prepared with a circuit
            #2. A unitary U is randomly selected from the ensemble and applied to \rho
            #3. A computational basis measurement is performed
            #4. The snapshot is recorded as the observed eigenvalue 1, -1 for |0>, |1>, respectively
            # and the index of the randomly selected

            # prepare the complete set of available Pauli operators
            # applying the single-qubit Clifford circuit is equivalent to measuring a Pauli
            unitary_ops = ["X", "Y", "Z"]
            # sample random Pauli measurements uniformly, where 0,1,2 = X,Y,Z
            # note that the information unitary_ensmb is important also (since it is the index of unitary)
            unitary_ensmb = np.random.randint(0, 3, size=(self.n_shots, n_wires_qpm), dtype=int)

            # we create the obs_all to obtain the mesurement record
            for ns in range(self.n_shots):
              obs_ls = []
              for i in range(n_wires_qpm):
                obs = []
                for j in range(n_wires_qpm):
                  if j == i:
                    obs.append(unitary_ops[unitary_ensmb[ns, i]])
                  else:
                    obs.append("I")
                obs_ls.append("".join(obs))
              obs_all.append(obs_ls)
          elif self.use_record == 2:
            # using Z basis measurement but with correlated Z to 
            # increase the number of features
            prod_ls = list(itertools.product('IZ', repeat = n_wires_qpm))
            for obs in prod_ls[1:]:
              obs_all.append("".join(obs))
          elif self.use_record == 3:
            # To increase measurement information, add observables. For 8 qubits, the total number of observables is 312.　{𝛼_𝑖}　{𝛼_𝑖 𝛽_(𝑖+1)} {𝛼_𝑖 𝛽_(𝑖+1) 𝛾_(𝑖+2)}　𝛼,𝛽,𝛾∈{𝑋,𝑌,𝑍}
            pauli_list = ["X", "Y", "Z"]
            for i in range(len(pauli_list)):
              for j in range(n_wires_qpm):
                obs = ['I'] * n_wires_qpm
                obs[j] = pauli_list[i]
                obs_all.append("".join(obs))

            for i in range(len(pauli_list)):
              for j in range(len(pauli_list)):
                for k in range(n_wires_qpm):
                  obs = ['I'] * n_wires_qpm
                  knext = (k+1)%n_wires_qpm
                  obs[k] = pauli_list[i]
                  obs[knext] = pauli_list[j]
                  obs_all.append("".join(obs))

            for i in range(len(pauli_list)):
              for j in range(len(pauli_list)):
                for k in range(len(pauli_list)):
                  for m in range(n_wires_qpm):
                    obs = ['I'] * n_wires_qpm
                    mnext = (m+1)%n_wires_qpm
                    mnextnext = (m+2)%n_wires_qpm
                    obs[m] = pauli_list[i]
                    obs[mnext] = pauli_list[j]
                    obs[mnextnext] = pauli_list[k]
                    obs_all.append("".join(obs))
                    
          else:
            # create a list of measurement operators, each measuring a single qubit in the Z basis.
            for i in range(n_wires_qpm):
                obs = ['I'] * n_wires_qpm
                obs[i] = 'Z'
                obs_all.append("".join(obs))

          self.obs_all = obs_all
          # print(f'self.nonlinear == quantum_data Pauli obs: {obs_all}')      
        
        # Additional gates to train qfm (but it is only enable when train_qfm > 0)
        if 'pars_' in self.nonlinear:
          self.variational_gates = [tqf.rx] * self.n_wires + [tqf.rz] * self.n_wires + [tqf.rx] * self.n_wires + [tqf.rzz] * self.n_wires
          self.variational_gates = self.variational_gates * self.var_depth
          

        if self.residual > 0:
          if self.train_qfm > 0:
            self.num_nonlinear_out = num_nonlinear_out + linear_out_fes
          else:
            self.num_nonlinear_out = num_nonlinear_out + linear_in_fes
        else:
          self.num_nonlinear_out = num_nonlinear_out
        
        # For training
        if force_set_lin_in or self.train_qfm > 0:
          lin_in = linear_in_fes
        else:
          lin_in = self.num_nonlinear_out
        
        self.P = nn.Linear(lin_in, linear_out_fes, bias=True) 
        print(f'P shape {lin_in}x{linear_out_fes} with nonlinear {self.nonlinear} out = {self.num_nonlinear_out}')
        if self.use_cuda:
          self.P = self.P.cuda()

        # For goodness evaluation
        pivot_out = linear_out_fes

        #if self.use_record > 0 and self.n_shots > 0:
        if False:
          pivot = torch.empty(1, pivot_out).uniform_(0, 1)
          pivot = torch.bernoulli(pivot)
        else:
          pivot = torch.normal(0, 1, size=(1, pivot_out))
          pivot = normalize(pivot)
        
        if self.use_cuda:
          # 10000 should be number of data used in cuda?
          pivot = (pivot.repeat(10000, 1)).cuda()
        
        self.pivot = pivot

        # Specify the optimizer
        self._set_opt()

    def _set_opt(self):
        # Specify the optimizer
        # params = list(self.P.parameters()) + list(self.Q.parameters())
        # self.opt = Adam(params, weight_decay=weight_decay, lr=lr)

        if self.train_qfm > 0 and 'pars_' in self.nonlinear:
          # only train the variational params if train_qfm > 0 and 'pars_' in nonlinear
          self.module_list = nn.ModuleList( [self.P, self.var_params] )
        else:
          self.module_list = nn.ModuleList( [self.P] ) 

        self.num_params = count_model_params(self.module_list)
        self.opt = Adam(self.module_list.parameters(), weight_decay=self.weight_decay, lr=self.lr)

        # if 'pars_' in self.nonlinear:
        #   self.opt = Adam(list(self.P.parameters()) + list(self.var_params), weight_decay=self.weight_decay, lr=self.lr)
        # else:
        #   self.opt = Adam(list(self.P.parameters()), weight_decay=self.weight_decay, lr=self.lr)
        
        
    def _gen_rand_params(self):
        self.P.weight.data.uniform_(self.thetamin, self.thetamax)
        # Reset the random params if we consider the variational model in quantum circuit
        if 'pars_' in self.nonlinear:
          # self.rx0.reset_params()
          # self.rx1.reset_params()
          # self.rz0.reset_params()
          # self.rzz0.reset_params()
          # self.var_params = nn.Parameter(torch.rand(4 * self.n_wires * self.var_depth) * 2 * np.pi)
          self.var_params = nn.ParameterList([
            nn.Parameter(torch.rand(1) * 2 * np.pi) for _ in range(4 * self.n_wires * self.var_depth)
          ])

        # Specify the optimizer
        self._set_opt()

    def forward(self, x):
      x = self.P(x)
      return x

    def variational_helper(self, qdev):
      for d in range(self.var_depth):
        bg = 4*self.n_wires*d
        for j in range(self.n_wires):
          jbg = j + bg
          self.variational_gates[jbg](qdev, wires=j, params=self.var_params[jbg]) #RX
          self.variational_gates[jbg + self.n_wires](qdev, wires=j,   params=self.var_params[jbg + self.n_wires]) #RZ
          self.variational_gates[jbg + 2*self.n_wires](qdev, wires=j, params=self.var_params[jbg + 2*self.n_wires]) #RX
        
        for j in range(self.n_wires):
          jbg = j + bg
          jnext = (j+1) % self.n_wires
          # RZZ
          self.variational_gates[jbg + 3*self.n_wires](qdev, wires=[j, jnext], params=self.var_params[jbg + 3*self.n_wires])

    def nonlinear_forward(self, x, input_x):
      # forward from digital network
      if ('qenc' in self.nonlinear) == False:
        x_out = self.act_func(x)
      else:
        if self.qenc_in_norm > 0:
          # Default not being used
          # Normalize before forwarding to quantum circuit
          x = torch.nn.Sigmoid()(x) * (2*torch.pi)

        # forward from quantum devices
        # create a quantum device to run the gates
        bsz, fsz = x.shape[0], x.shape[1]

        n_c = int(fsz/self.n_fet_enc)
        n_all_fet_out = self.n_fet_out * n_c
        
        if n_all_fet_out >= x.shape[1]:
          x_out = nn.ConstantPad1d((0, n_all_fet_out - x.shape[1]), 0.0)(x)
        else:
          x_out = x[:, :n_all_fet_out].clone()

        # encode the classical information to quantum domain
        for n in range(n_c):
          qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device)

          if self.rc == 1: # residual connection
            # Set the input wave function input_x as the state of the QuantumDevice.
            qdev.set_states(input_x) 
            qdev.states = qdev.states.to(torch.complex64)  # Convert the data type of the quantum state.

          # Repeat encoding (reuploading)
          for _ in range(self.rep_enc):
            if 'qenc_44iqp_full' in self.nonlinear:
              for k in range(4): # 4 layers of ZZFeatureMap
                for j in range(self.n_wires):
                  self.Hadamard(qdev, wires=j)
                  self.encoder_gates[j](qdev, wires=j, params=x[:, k*self.n_wires + j + n * self.n_fet_enc])
                for j in range(self.n_wires):
                  jnext = (j+1)%self.n_wires
                  p1 = x[:, k*self.n_wires + j + n * self.n_fet_enc]
                  p2 = x[:, k*self.n_wires + jnext + n * self.n_fet_enc]
                  self.encoder_gates[j + self.n_wires](qdev, wires=[j, jnext], params=p1*p2 )
            else:
              self.encoder(qdev, x[:, (n*self.n_fet_enc) : ((n+1)*self.n_fet_enc)])

            if 'pars_' in self.nonlinear:
              # add some trainable gates (need to instantiate ahead of time, NOTE: should be only for train_qfm > 0 but not in current implementation)
              # work as preprocessing layer if train_qfm = 0
              self.variational_helper(qdev)

            elif 'rand_' in self.nonlinear:
              # Add some random layers
              self.random_layers[n](qdev)
            
            # for multi-bases measurement
            # random rotate and measure in different basis
            for nb in range(0, self.n_basis):
              # for each group need to clone a new qdev and its states
              if self.n_basis == 1:
                qdev_clone = qdev
              else:
                qdev_clone = tq.QuantumDevice(n_wires=qdev.n_wires, bsz=qdev.bsz, device=qdev.device)
                qdev_clone.clone_states(qdev.states)
              if nb > 0:
                for wire in range(self.n_wires):
                  self.random_rotXs[nb-1](qdev_clone, wire)

              # perform measurement to get expectations (back to classical domain)
              if self.use_record == 1 and self.n_shots > 0:
                # # Measure the state on the z basis
                # ord_ls = tq.measure(qdev, n_shots=self.n_shots)
                # img_shot = []
                # for k, ord in enumerate(ord_ls):
                #   bstring = ''.join([bs*f for bs, f in ord.items()])
                #   bvec = np.array([int(b) for b in bstring]) 
                #   img_shot.append(bvec)
                # img_shot = np.array(img_shot)
                # res = torch.from_numpy(img_shot.astype(np.float32)).clone()

                # Use shadow
                fet_ls = []
                for obs_ls in self.obs_all:
                    expval_sam = expval_joint_sampling_grouping(qdev_clone, observables=obs_ls, n_shots_per_group=1)
                    for obs in obs_ls:
                      expval = expval_sam[obs]
                      fet_ls.append(expval)
                res = torch.stack(fet_ls).T

                if self.use_cuda:
                  res = res.cuda()
              elif self.n_obs > 0:
                fet_ls = []
                if self.n_shots == 0:
                  for obs in self.obs_all:
                      expval = expval_joint_analytical(qdev_clone, observable=obs)
                      fet_ls.append(expval)
                  res = torch.stack(fet_ls).T
                else:
                  # Finite shots measurement
                  expval_sam = expval_joint_sampling_grouping(qdev_clone, observables=self.obs_all, n_shots_per_group=self.n_shots)
                  for obs in self.obs_all:
                      expval = expval_sam[obs]
                      fet_ls.append(expval)
                  res = torch.stack(fet_ls).T
              else:
                # use_record == 0
                if self.n_shots == 0:
                  res = self.measure(qdev_clone)
                else:
                  # Finite shots measurement
                  fet_ls = []
                  expval_sam = expval_joint_sampling_grouping(qdev_clone, observables=self.obs_all, n_shots_per_group=self.n_shots)
                  for obs in self.obs_all:
                      expval = expval_sam[obs]
                      fet_ls.append(expval)
                  res = torch.stack(fet_ls).T
              
              if self.train_qfm > 0:
                # Normalize the nonlinear output, 
                # this normalization may be important to stabilize the training
                res =  qenc_out_normalize(res * self.scale, self.qenc_out_norm)
              
              res_sz = res.shape[1]
              x_out[:, (n*self.n_fet_out + nb*res_sz):(n*self.n_fet_out + (nb+1)*res_sz)] = res
      
      # concat with previous features (for residual)
      if self.residual > 0:
        x_out = torch.cat((x,x_out), 1)
      return x_out
    
    # Input: quantum data
    def nonlinear_forward_qpm(self, x):
      # TODO: shadows and limited shots
      # forward from quantum devices
      # create a quantum device to run the gates

      bsz, fsz = x.shape[0], x.shape[1]

      # QuantumDevice Initialize the object.
      n_wires = self.n_wires
      qdev = tq.QuantumDevice(n_wires=n_wires, bsz=bsz, device=x.device) 
      qdev.set_states(x) 
      qdev.states = qdev.states.to(torch.complex64)  # Convert the data type of the quantum state.
      n_all_fet_out = self.linear_in_fes


      if n_all_fet_out > x.size(1): # Initialize a new tensor with zeros.
        x_out = torch.zeros(x.size(0), n_all_fet_out)
      else:
        x_out = x[:, :n_all_fet_out].clone()


      for _ in range(1):
        if self.noise_level > 0 : # Add noise to the data.
          # Apply noise that randomly rotates in the x-axis direction on the first layer.
          # Make each data have a random rotation angle.
          random_angles = np.random.uniform(0, 2 * np.pi, size=n_wires) * self.noise_level
          for j in range(n_wires):
            self.noise_gates[j](qdev, wires=j, params=random_angles[j])

        # Add Hadamard gate
        for j in range(n_wires):
         self.Hadamard(qdev, wires=j)

        # Add random ZZ gates after the quantum data
        for d in range(self.var_depth):
          bg = d * 2 * self.n_wires
          for j in range(n_wires):
            self.encoder_gates[j](qdev, wires=[j], params=self.rand_enc_params[j + bg])

          for j in range(n_wires):
            jnext = (j+1)%self.n_wires
            self.encoder_gates[j + n_wires](qdev, wires=[j, jnext], params=self.rand_enc_params[j + bg + n_wires])

        if 'pars_' in self.nonlinear: 
          # add some trainable gates (need to instantiate ahead of time)
          self.variational_helper(qdev)

        # Repeat encoding (reuploading)
        for _ in range(self.rep_enc): 
          # for multi-bases measurement
          # random rotate and measure in different basis
          for nb in range(0, self.n_basis):
            # for each group need to clone a new qdev and its states
            if self.n_basis == 1:
              qdev_clone = qdev # Confirm the reason for the clone.
              qdev_clone.states = qdev_clone.states.to(torch.complex64) # Check later if it's okay not to use complex128.
            else:
              qdev_clone = tq.QuantumDevice(n_wires=qdev.n_wires, bsz=qdev.bsz, device=qdev.device)
              qdev_clone.clone_states(qdev.states)
              qdev_clone.states = qdev_clone.states.to(torch.complex64)
            if nb > 0:
              # print("nonlinear_forward_qpm nb > 0 nb=", nb)
              for wire in range(n_wires):
                self.random_rotXs[nb-1](qdev_clone, wire)
            
            # perform measurement to get expectations (back to classical domain)
            if self.use_record == 1 and self.n_shots > 0:
              fet_ls = []
              for obs_ls in self.obs_all:
                  expval_sam = expval_joint_sampling_grouping(qdev_clone, observables=obs_ls, n_shots_per_group=1)
                  for obs in obs_ls:
                    expval = expval_sam[obs]
                    fet_ls.append(expval)
              res = torch.stack(fet_ls).T

              if self.use_cuda:
                res = res.cuda()
            elif self.n_obs > 0:
              fet_ls = []
              if self.n_shots == 0:
                for obs in self.obs_all:
                    # Match the data types of quantum states and Hamiltonians.
                    expval = expval_joint_analytical(qdev_clone, observable=obs)
                    fet_ls.append(expval)
                res = torch.stack(fet_ls).T
              else:
                expval_sam = expval_joint_sampling_grouping(qdev_clone, observables=self.obs_all, n_shots_per_group=self.n_shots)
                for obs in self.obs_all:
                    expval = expval_sam[obs]
                    fet_ls.append(expval)
                res = torch.stack(fet_ls).T
            else:
              res = self.measure(qdev_clone)
            
            if self.train_qfm > 0:
              # Normalize the nonlinear output, 
              # this normalization may be important to stabilize the training
              res =  qenc_out_normalize(res * self.scale, self.qenc_out_norm)
            
            res_sz = res.shape[1]
            x_out[:, (nb*res_sz):((nb+1)*res_sz)] = res

      # concat with previous features (for residual)
      if self.residual > 0:
        x_out = torch.cat((x,x_out), 1)
      return x_out
    
    def good_helper(self, x, anchor, type_pivot):
      """
      Helper to calculate the goodness function
      type_pivot: 0 for fixed pivot, 1 for random pivot selected from data
      """
      # STEP 2: classical forward and normalize
      h = (self.forward(x))
      anchor = (self.forward(anchor))
      h = qenc_out_normalize(h, self.qenc_out_norm)
      anchor = qenc_out_normalize(anchor, self.qenc_out_norm)
      if type_pivot == 0:
        g = sim_func(h, self.pivot)
      else:
        g = sim_func(h, anchor)
      return g

    # calculate the goodness function
    def goodness(self, x_pos, x_neg, anchor, type_pivot):
      g_pos = self.good_helper(x_pos, anchor, type_pivot)
      g_neg = self.good_helper(x_neg, anchor, type_pivot)
      return g_pos, g_neg

    def single_quantum_forward_helper(self, x, input_x):
      if 'quantum_data' in self.nonlinear:
        h = self.nonlinear_forward_qpm(x)
      else:
        h = self.nonlinear_forward(x, input_x)
      h = h.real.to(torch.float32)
      return h
    
    def quantum_forward_helper(self, x_pos, x_neg, anchor, input_x_pos, input_x_neg, input_anchor):
        h_pos = self.single_quantum_forward_helper(x_pos, input_x_pos)
        h_neg = self.single_quantum_forward_helper(x_neg, input_x_neg)
        anchor = self.single_quantum_forward_helper(anchor, input_anchor)
        return h_pos, h_neg, anchor

    def info_nce_loss(self, features, temperature=0.5):
        """
        Calculate the InfoNCE loss for a batch of features.
        """
        # features: (2N, D) where N is the batch size and D is the feature dimension
        # Assume the first N are positive pairs with the second N
        N = features.shape[0] // 2
        z_i, z_j = features[:N], features[N:]

        # Normalize the feature vectors
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # Compute similarity scores
        similarity_matrix = torch.mm(z_i, z_j.t()) / temperature

        # Create labels for positive pairs
        labels = torch.arange(N).to(features.device)

        # Compute the loss
        loss = F.cross_entropy(similarity_matrix, labels)
        return loss
    
    # train function (qpm version)
    def train_qpm(self, x_pos, x_neg, anchor, input_x_pos, input_x_neg, input_anchor, type_pivot):
        self.train_loss=0.0
        #print("self.nonlinear", self.nonlinear)
        prev_anchor = anchor.clone()

        # IF TRAIN_QFM=0, we only need to train classical forward in inner loop
        if self.train_qfm == 0:
          h_pos, h_neg, anchor = self.quantum_forward_helper(x_pos, x_neg, prev_anchor, input_x_pos, input_x_neg, input_anchor)

        #scale = self.scale * self.num_nonlinear_out
        scale = self.scale

        for i in range(self.n_epochs_inner):
          #print(f'i={i}, train qfm={self.train_qfm}, anchor_shape={anchor.shape}')

          # IF TRAIN_QFM>0, we need to train both the quantum and classical forward in inner loop
          if self.train_qfm > 0:
           h_pos, h_neg, anchor = self.quantum_forward_helper(x_pos, x_neg, prev_anchor, input_x_pos, input_x_neg, input_anchor)

          if self.type_anchor == 1:
            g_pos, g_neg = self.goodness(h_pos, h_neg, anchor, type_pivot)
            delta = g_pos - g_neg
            # Calculation of the loss function
            if self.type_cost == 'rbf':
              loss  = (torch.log(1 + torch.exp(-scale*delta))).mean()
            else:
              loss = (-delta).mean()
          elif self.type_anchor == 0:
            # Concatenate positive and negative features for InfoNCE loss
            features = torch.cat([h_pos, h_neg], dim=0)

            # Calculate InfoNCE loss
            loss = self.info_nce_loss(features)

          self.opt.zero_grad() # Initialization of gradients
          #print(f'{self.nonlinear}-{i} Calling backward')
          
          loss.backward(retain_graph=True) # Calculate the gradient based on the loss function (propagation).
          self.opt.step() # Update the model parameters using the computed gradients.
          self.train_loss += loss.item() # Cumulative training loss
        h_pos = h_pos.real.to(torch.float32)
        h_neg = h_neg.real.to(torch.float32)
        anchor = anchor.real.to(torch.float32)

        
        # STEP 2: including the linear forward and normalized layer
        h_pos, h_neg, anchor = (self.forward(h_pos)), (self.forward(h_neg)), (self.forward(anchor))

        # Normalization function
        h_pos = qenc_out_normalize(h_pos, self.qenc_out_norm)
        h_neg = qenc_out_normalize(h_neg, self.qenc_out_norm)
        anchor = qenc_out_normalize(anchor, self.qenc_out_norm)

        #print(f'Inner {i} at {self.train_loss}')
        return h_pos.detach(), h_neg.detach(), anchor.detach(), self.train_loss/self.n_epochs_inner
    

class DeepNet(Model, torch.nn.Module):
    def __init__(self, dims, n_epochs_outer, n_epochs_inner, n_labels,
          weight_decay=0, lr=0.0001, rc=0,scale=1.0, n_wires=8, n_shots=0, use_record=0, n_basis=1,
          noise_level=0, n_obs=0, use_BP=0, rep_enc=1, residual=0, type_cost='rbf', type_anchor='rand', train_qfm=0, qenc_in_norm=0, qenc_out_norm=0,
          use_cuda=False, nonlinear=GELU, logger=None, num_tmp=100, var_depth=2, gather_features=True): # Change n_wires to 8.
        Model.__init__(self)
        torch.nn.Module.__init__(self)

        self.layers = []
        self.module_list = nn.ModuleList([])
        self.num_params = 0
        self.para_weight = []
        self.para_bias = []
        self.loss_save = defaultdict(list)
        self.param_save = []
        self.n_epochs_outer = n_epochs_outer
        self.n_labels = n_labels
        self.N_layer  = int(len(dims)) - 1
        self.loss_list = [0] * self.N_layer
        self.use_cuda = use_cuda
        self.nonlinear = nonlinear
        self.type_cost = type_cost
        self.type_anchor = type_anchor
        self.train_qfm = train_qfm
        self.qenc_in_norm = qenc_in_norm
        self.qenc_out_norm = qenc_out_norm
        self.num_tmp = num_tmp
        self.n_wires = n_wires
        
        # The features are collected at every layer (TRUE) or only the final layer (FALSE)
        self.gather_features = gather_features

        self.logger = logger
        if self.logger is not None:
          self.save_dir, log_file = get_logger_parent_folder(self.logger)
          self.save_dir = os.path.join(self.save_dir, log_file.replace('.log', ''))
          self.base_name = ''
        else:
          self.save_dir = '../temp_res'
          self.base_name = 'temp'
        
        lin_in = dims[0]
        # self.logger.info(f'self.dims: {dims}, self.N_layer {self.N_layer} lin_in={lin_in}')
        force_set_lin_in = False
        for d in range(0, self.N_layer):
            layer_nonlinear = self.nonlinear
            if 'qenc' in self.nonlinear and d == 0: #1layer目
              # layer_nonlinear = 'id'
              # Input the wave function.
              layer_nonlinear = 'quantum_data'
              if 'pars_' in self.nonlinear:
                # variational model even for the first layer
                layer_nonlinear = 'pars_quantum_data'
            
            lay = Layer(lin_in, dims[d+1], n_epochs_inner=n_epochs_inner, force_set_lin_in=force_set_lin_in,\
                    weight_decay=weight_decay, lr=lr, rc=rc, scale=scale, n_wires=n_wires, use_record=use_record, n_shots=n_shots, \
                    qenc_in_norm=qenc_in_norm, qenc_out_norm=qenc_out_norm,\
                    n_basis=n_basis, use_BP=use_BP, rep_enc=rep_enc, train_qfm=train_qfm, residual=residual, type_cost=type_cost, \
                    type_anchor=type_anchor, noise_level=noise_level, n_obs=n_obs, use_cuda=self.use_cuda, nonlinear=layer_nonlinear, var_depth=var_depth)
            if force_set_lin_in:
              # Only set linear_in one time
              force_set_lin_in = False
            if self.logger is not None:
              self.logger.info(f'Linear layer{d}: {lin_in}x{dims[d+1]}, nonlinear {layer_nonlinear} out={lay.num_nonlinear_out}')
            if train_qfm == 0:
              if 'qenc' in self.nonlinear:
                if d == 0:
                  # Set lin_in manual since the first layer is quantum_data
                  if self.logger is not None:
                    self.logger.info(f'd == 0 lin_in : {lin_in}')
                    self.logger.info(f'd == 0 n_wires : {n_wires}')
                    self.logger.info(f'd == 0 n_basis : {n_basis}')
                  self.n_fet_enc = self.n_wires * 4
                  lin_in = get_num_out_features(dims[d+1], self.nonlinear, n_wires, n_obs, n_shots, use_record, self.n_fet_enc, n_basis=n_basis)  

                  if residual > 0:
                    lin_in = lin_in + dims[d+1]
                  force_set_lin_in = True
                else:
                  lin_in = lay.num_nonlinear_out
              else:
                lin_in = dims[d+1]
            else:
              lin_in = lay.num_nonlinear_out
            self.logger.info(f'lin_in: {lin_in}')
            
            if self.use_cuda:
               lay = lay.cuda()
            #print('Lay', d, lay.num_params)
            self.layers += [lay]
            self.module_list += [lay.module_list]
            
        self.num_params = count_model_params(self.module_list)
        print('Num params in Deeplayer', self.num_params)
        self.input_dim = dims[0]

    def _gen_rand_params(self):
      for lay in self.layers:
        lay._gen_rand_params()

    def forward(self, x, params=None):
        """
        Computes the output of the IQFM
        :param x: ndarray, data inputs for the model (can be one or multiple)
        :param params (default=None): here, random params are used, need to add functionality for using passed params
        :return: torch tensor, model output of size (len(x), output_size)
        """
        if not torch.is_tensor(x):
          x = torch.from_numpy(x)
        if self.use_cuda:
          x = x.cuda()
        for k, layer in enumerate(self.layers):
          if layer.train_qfm == 0:
            # nonlinear first if we do not need to train qfm
            x = layer.nonlinear_forward(x)
          x = layer.forward(x)
          if layer.train_qfm > 0:
            # if train qfm, we must put the nonlinear forward
            x = layer.nonlinear_forward(x)
          else:
            x = qenc_out_normalize(x, layer.qenc_out_norm)
        x = F.softmax(x, dim=-1)
        return x
    
    def forward_qpm(self, x, params=None):
        """
        Computes the output of the IQFM
        :param x: ndarray, data inputs for the model (can be one or multiple)
        :param params (default=None): here, random params are used, need to add functionality for using passed params
        :return: torch tensor, model output of size (len(x), output_size)
        """
        # if not torch.is_tensor(x): 
        #   x = torch.from_numpy(x)
        # if self.use_cuda:
        #   x = x.cuda()
        for k, layer in enumerate(self.layers):
          if layer.train_qfm == 0:
            # nonlinear first if we do not need to train qfm
            x = layer.nonlinear_forward_qpm(x)
          x = layer.forward(x)
          if layer.train_qfm > 0:
            # if train qfm, we must put the nonlinear forward
            x = layer.nonlinear_forward_qpm(x)
          else:
            x = qenc_out_normalize(x, layer.qenc_out_norm)
        x = F.softmax(x, dim=-1)
        return x

    def get_gradient(self, x, params=None):
      """
        Computes the gradients of every parameter using each input x, wrt every output.
        :param x: ndarray, data inputs for the model (can be one or multiple)
        :param params: here, random params are used, need to add functionality for using passed params
        :return: numpy array, gradients of size (len(x), output_size, d)
      """
      if not torch.is_tensor(x):
          x = torch.from_numpy(x)
          x.requires_grad_(False)
      grad_vecs = []
      seed = 0
      for i in range(len(x)):
        if i % self.num_tmp == 0:
          seed += 1
        torch.manual_seed(seed)
        self._gen_rand_params()
        output = self.forward(x[i].unsqueeze(0), params)[0]
        log_output = torch.log(output)  # get the output values to calculate the jacobian
        grad = []
        output_size = len(output)
        for j in range(output_size):
          self.zero_grad()
          log_output[j].backward(retain_graph=True)
          grads = []
          for param in self.module_list.parameters():
            grads.append(param.grad.view(-1))
          #print(self.params)
          gr = torch.cat(grads)
          grad.append(gr * torch.sqrt(output[j]))
        jacobian = torch.cat(grad)
        #print(jacobian.shape, self.num_params)
        jacobian = torch.reshape(jacobian, (output_size, self.num_params))
        grad_vecs.append(jacobian.detach().numpy())
      return grad_vecs
    
    def get_fisher(self, gradients):
        """
        Computes average gradients over outputs.
        :param gradients: numpy array containing gradients
        :return: numpy array, average jacobian of size (len(x), d)
        """
        output_size = gradients[0].shape[0]
        fishers = np.zeros((len(gradients), self.num_params, self.num_params))
        for i in range(len(gradients)):
          grads = gradients[i]
          tmp_sum = np.zeros((output_size, self.num_params, self.num_params))
          for j in range(output_size):
            tmp_sum[j] += np.array(np.outer(grads[j], np.transpose(grads[j])))
          fishers[i] += np.sum(tmp_sum, axis=0)
          #print('Fisher', i, grads, fishers[i])
        return fishers
    
    # A function to obtain feature vectors using the model after training.
    # Function for finding feature vector using trained model
    def get_feature_vector(self, h):
        with torch.no_grad():
          # A list storing the feature vector g
          g_list = []
          input_h = h

          for _, layer in enumerate(self.layers):
            if 'quantum_data' in layer.nonlinear: # Input of the first wave function
              h = (layer.nonlinear_forward_qpm(h))
              h = h.real.to(torch.float32)
            else:
              h = (layer.nonlinear_forward(h, input_h))

            # (FORGET about it) Tran(12/25) Since the forward will discard some info I recommend to use the 
            # previous features to learn, but use the forwared feature to calculate the goodness
            # Enable the following code if you want to use the forwared feature to learn in classifier

            h = (layer.forward(h))
            h = qenc_out_normalize(h, self.qenc_out_norm)

            g = (h)
            g_list.append(g)

          if self.gather_features:
            # Concatenate feature vectors along the column direction.
            res_g = torch.cat(g_list, dim=1)
          else:
            # Only using the feature of the last layer
            res_g = g_list[-1]
        return res_g
    
    def build_feature_loaders(self, train_loader_qpm, test_loader_qpm):
        """
        From the QPM data loader, precompute features once,
        and convert it to a DataLoader that returns (features, labels).
        Do not call get_feature_vector thereafter.
        """
        def _extract_features(loader):
            feats, tgts = [], []
            with torch.no_grad():
                for x, y in loader:
                    g = self.get_feature_vector(x)   # Pre-calculation is only here
                    feats.append(g.cpu())
                    tgts.append(y.cpu())
            feats = torch.cat(feats, dim=0)
            tgts  = torch.cat(tgts,  dim=0)
            ds = TensorDataset(feats, tgts)
            return ds

        train_feat_ds = _extract_features(train_loader_qpm)
        test_feat_ds  = _extract_features(test_loader_qpm)

        train_bs = train_loader_qpm.batch_size
        test_bs  = test_loader_qpm.batch_size

        train_feat_loader = DataLoader(train_feat_ds, batch_size=train_bs, shuffle=True)
        test_feat_loader  = DataLoader(test_feat_ds,  batch_size=test_bs,  shuffle=False)

        return train_feat_loader, test_feat_loader
    
    def train_classifier(self, train_feat_loader, test_feat_loader, num_epochs=500):
        acc_test_2=[]
        acc_train_2=[]

        # 1) Get dimensions and number of classes from precomputed features
        for j, data in enumerate(train_feat_loader, 0):
          feats, labels = data
          input_dim = feats.size(1)
          unique_labels = labels.unique().tolist()
          num_classes = len(unique_labels)

          # # PCA analysis
          # n_components = 2
          # pca = PCA(n_components=n_components)
          # pca_result = pca.fit_transform(combined_g)

          # # UMAP
          # umap_model = umap.UMAP(n_components=n_components)
          # umap_result = umap_model.fit_transform(combined_g)
          
          # # Save the results to a log file.
          # if self.logger is not None:
          #   self.logger.info(f'pca_result:{pca_result.tolist()}')
          #   self.logger.info(f'pca_explained_variance_ratio:{pca.explained_variance_ratio_.tolist()}')
          #   self.logger.info(f'umap_result:{umap_result.tolist()}')   
          #   self.logger.info(f'input_dim:{input_dim}') 
          #   # self.logger.info(f'layer_dim:{layer_dim}')  
          #   self.logger.info(f'train_labels_qpm:{train_labels_qpm}')    
          
          break


        # 2) Classifier (final layer without Softmax: CE assumes logits)
        classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim), 
            nn.ReLU(),               
            nn.Linear(input_dim, input_dim), 
            nn.ReLU(),                
            nn.Linear(input_dim, num_classes),
        )
        criterion = nn.CrossEntropyLoss()
        
        def count_parameters(model):
          return sum(p.numel() for p in model.parameters() if p.requires_grad)

        if self.use_cuda:
            classifier = classifier.cuda()

        optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001, weight_decay=1e-4)

        if self.logger is not None:
            classifier_num_params = count_parameters(classifier)
            self.logger.info(f'Classifier input_dim={input_dim}, num_classes={num_classes}, num_params={classifier_num_params}')
            self.logger.info(f'Classifier model: {classifier}')
        
        for epoch in range(num_epochs):
            classifier.train()  # Train mode
            for feats, labels in train_feat_loader:
              if self.use_cuda:
                  feats, labels = feats.cuda(), labels.long().cuda()
              else:
                  labels = labels.long()

              outputs = classifier(feats)
              loss = criterion(outputs, labels)

              optimizer.zero_grad()
              loss.backward()
              optimizer.step()

            # Evaluate test loss every 10 epochs or at epoch 0
            if epoch == 0 or epoch % 10 == 9:
              train_loss, train_acc = self.eval_acc_classifier(criterion, classifier, train_feat_loader, unique_labels)
              test_loss, test_acc = self.eval_acc_classifier(criterion, classifier, test_feat_loader, unique_labels)
              train_acc *= 100
              test_acc *= 100                

              acc_train_2.append(train_acc)
              acc_test_2.append(test_acc)

              if self.logger is not None:
                self.logger.info(f'[(Classifier) Epoch={epoch + 1}], train loss={train_loss:.6f}, test loss={test_loss:.6f}, train accuracy={train_acc:.2f}, test accuracy={test_acc:.2f}')
        
    def train_regressor(self, combined_train_loader_qpm, combined_test_loader_qpm, num_epochs=500):
        train_losses = []
        test_losses = []

        # 1. Determine feature vector dimension from a sample batch
        for j, data in enumerate(combined_train_loader_qpm, 0):
            train_data_qpm, train_target_qpm = data
            combined_g = self.get_feature_vector(train_data_qpm)
            
            input_dim = combined_g.size(1)  # Feature vector dimension
            break

        # 2. Define the regressor model
        regressor = nn.Sequential(
            nn.Linear(input_dim, input_dim), 
            nn.ReLU(),                        
            nn.Linear(input_dim, input_dim),  
            nn.ReLU(),                        
            nn.Linear(input_dim, 1)           
        )
        criterion = nn.MSELoss()  # Use MSE loss for regression

        if self.use_cuda:
            regressor = regressor.cuda()

        optimizer = torch.optim.Adam(regressor.parameters(), lr=0.001, weight_decay=1e-4)

        # We do not use early stopping for regression
        # Early stopping parameters
        # patience = 10  # Number of epochs to wait for improvement
        # best_loss = float('inf')  # Track the lowest test loss
        # patience_counter = 0  # Count epochs without improvement

        for epoch in range(num_epochs):
            regressor.train()  # Train mode
            for j, data in enumerate(combined_train_loader_qpm, 0):
                train_data_qpm, train_target_qpm = data
                combined_g = self.get_feature_vector(train_data_qpm)

                train_data = combined_g
                train_target = train_target_qpm.float()  # Ensure target is float
                # print(train_target)
                
                # Forward pass
                outputs = regressor(train_data)
                loss = criterion(outputs.squeeze(), train_target)  # Squeeze to match dimensions

                # Backpropagation and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Display loss every epoch
            if self.logger is not None:
                self.logger.info(f'[(Regressor) Epoch={epoch + 1}], train loss={loss.item():.6f}')
            
            # Evaluate test loss every 10 epochs or at epoch 0
            if epoch == 0 or epoch % 10 == 9:
                train_loss = self.eval_loss(criterion, regressor, combined_train_loader_qpm)
                test_loss = self.eval_loss(criterion, regressor, combined_test_loader_qpm)

                train_losses.append(train_loss)
                test_losses.append(test_loss)

                if self.logger is not None:
                    self.logger.info(f'[(Regressor) Epoch={epoch + 1}], train loss={train_loss:.6f}, test loss={test_loss:.6f}')

                # We do not use early stopping for regression
                # # Early stopping logic based on test loss
                # if test_loss < best_loss:
                #     best_loss = test_loss
                #     patience_counter = 0  # Reset patience counter
                # else:
                #     patience_counter += 1

                # if patience_counter >= patience:
                #     if self.logger is not None:
                #         self.logger.info(f'Early stopping triggered at epoch {epoch + 1}. Best test loss={best_loss:.6f}')
                #     break

        return train_losses, test_losses


    def eval_acc_classifier(self, criterion, classifier, data_loader, unique_labels):
        classifier.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for feats, labels in data_loader:        
                if self.use_cuda:
                    feats, labels = feats.cuda(), labels.long().cuda()
                else:
                    labels = labels.long()

                logits = classifier(feats)
                loss = criterion(logits, labels)
                probs = torch.softmax(logits, dim=1)
                predicted_class = probs.argmax(dim=1)
                correct = (predicted_class == labels).sum().item()

                bs = labels.size(0)
                total_loss   += loss.item() * bs
                total_correct += correct
                total_samples += bs

        avg_loss = total_loss / total_samples
        avg_acc  = total_correct / total_samples
        return avg_loss, avg_acc
    
    def eval_loss(self, criterion, model, data_loader):
      """
      Evaluate the loss on the given data loader.
      Args:
          criterion: Loss function.
          model: Regressor model.
          data_loader: DataLoader for the dataset.
      Returns:
          float: Average loss over the dataset.
      """
      model.eval()
      total_loss = 0.0
      num_samples = 0

      with torch.no_grad():
          for data in data_loader:
              inputs, targets = data
              combined_g = self.get_feature_vector(inputs)
              targets = targets.float()

              outputs = model(combined_g)
              loss = criterion(outputs.squeeze(), targets)
              total_loss += loss.item() * inputs.size(0)
              num_samples += inputs.size(0)

      return total_loss / num_samples

    def visualize_layers(self, epoch=None, ftypes=['png']):
      for k, layer in enumerate(self.layers):
        visualize_linear_layer(layer, f'{self.base_name}_layer_{k}', epoch, save_dir=self.save_dir, ftypes=ftypes)


    # Train by contrastive learning in the quantum phase transition classification task
    def train_qpm(self, combined_train_loader_qpm, combined_test_loader_qpm, type_task='classification', plot_layer=False, type_samples='rand'):
        # Visualize the layers at initial state
        if plot_layer:
          self.visualize_layers(epoch=0, ftypes=['png', 'pdf', 'svg'])
        if 'tensor' in type_samples:
          # We evaluate the trained model on train and test data
          train_acc = self.eval_acc(combined_train_loader_qpm) * 100.0
          test_acc = self.eval_acc(combined_test_loader_qpm) * 100.0
          if self.logger is not None:
            self.logger.info(f'[Start] Train accuracy={train_acc:.2f}, test accuracy={test_acc:.2f}')

        for epoch_outer in range(self.n_epochs_outer):
            self.loss_list=[0] * self.N_layer

            for j, data in enumerate(combined_train_loader_qpm, 0):
              train_data_qpm, train_labels_qpm = data
              
              # We create positive and negative samples for contrastive learning
              h_pos, h_neg = make_pos_neg_samples(train_data_qpm, train_labels_qpm, type=type_samples, num_labels = self.n_labels)

              input_h_pos = h_pos
              input_h_neg = h_neg
              anchor = train_data_qpm
              type_pivot = 1
              if 'tensor' in type_samples:
                anchor = extend_data_with_tensor_product(train_data_qpm, train_labels_qpm, self.n_labels)
                type_pivot = 0
              input_anchor = anchor

              # Train the model
              for k, layer in enumerate(self.layers):
                h_pos, h_neg, anchor, loss = layer.train_qpm(h_pos, h_neg, anchor, input_h_pos, input_h_neg, input_anchor, type_pivot)
                self.loss_list[k] = self.loss_list[k] + loss

              if False:
                # Plot layers at each epoch
                self.visualize_layers(epoch=epoch_outer+1)

            self.loss_list = np.divide(self.loss_list, j + 1)

            for l in range(self.N_layer):
              self.loss_save[l].append(self.loss_list[l])
            
            if 'tensor' in type_samples:
              # We evaluate the trained model on train and test data
              train_acc = self.eval_acc(combined_train_loader_qpm) * 100.0
              test_acc = self.eval_acc(combined_test_loader_qpm) * 100.0
              if self.logger is not None:
                self.logger.info(f'[Epoch={epoch_outer + 1}], train accuracy={train_acc:.2f}, test accuracy={test_acc:.2f}')

            if self.logger is not None:
              loss_str = ' '.join([str(s) for s in self.loss_list])
              self.logger.info(f'[Epoch={epoch_outer + 1}], loss=[{loss_str}]')

        # Plot layers at final
        if plot_layer:
          self.visualize_layers(ftypes=['png', 'pdf', 'svg'])

        if 'tensor' in type_samples:
          # We evaluate the trained model on train and test data
          train_acc = self.eval_acc(combined_train_loader_qpm) * 100.0
          test_acc = self.eval_acc(combined_test_loader_qpm) * 100.0
          if self.logger is not None:
            self.logger.info(f'[Final] Train accuracy={train_acc:.2f}, test accuracy={test_acc:.2f}')
        else:
          # Train for the second step: Since this part is computed only on classical computer
          # we can take a large number of epochs
          num_epochs = 500
          if type_task == 'reg_entropy':
            self.logger.info(f'Training regressor with entanglement entropy task') 
            reg_train_loader_qpm = preprocess_entanglement_entropy(combined_train_loader_qpm, n_subsystem=self.n_wires//2, batch_size=20, shuffle=True)
            test_train_loader_qpm = preprocess_entanglement_entropy(combined_test_loader_qpm, n_subsystem=self.n_wires//2, batch_size=20, shuffle=False)
            self.train_regressor(reg_train_loader_qpm, test_train_loader_qpm, num_epochs=num_epochs)
          else:
            self.logger.info(f'Training classifier')
            train_feat_loader, test_feat_loader = self.build_feature_loaders(
                combined_train_loader_qpm, combined_test_loader_qpm,
            )
                    
            self.train_classifier(train_feat_loader, test_feat_loader, num_epochs=num_epochs)
    
    def predict(self, x):
      """ Predict the class of the input data x """
      goodness_per_label = []
      for label in range(self.n_labels):
          h = over_data_with_tensor_product(x, label, self.n_labels)
          input_h = h.clone()
          goodness = []
          for _, layer in enumerate(self.layers):
            h = (layer.single_quantum_forward_helper(h, input_h))
            h = (layer.forward(h))
            h = qenc_out_normalize(h, self.qenc_out_norm)
            g = (h)
            goodness += [sim_func(g, layer.pivot)]

          goodness_per_label += [sum(goodness).unsqueeze(1)]
      goodness_per_label = torch.cat(goodness_per_label, 1)
      return goodness_per_label.argmax(1)
    
    def eval_acc(self, data_loader):
      acc = 0.0
      for g, data in enumerate(data_loader, 0):
        x_t, y_t = data
        if self.use_cuda:
          x_t, y_t = x_t.cuda(), y_t.cuda()
        acc += ( self.predict(x_t).eq(y_t).float().mean().item())
      return (acc)/(g+1)



if __name__ == "__main__":

  # Check for command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--save_dir', type=str, default='../results/test_iqfm_eff')

  # For network
  parser.add_argument('--nonlinear', type=str, default='qenc_44rxyzx', help='gelu,relu,qenc0,qenc1,qenc2')
  parser.add_argument('--rep_enc', type=int, default=1)
  parser.add_argument('--residual', type=int, default=0, help='Use residual structure or not')
  parser.add_argument('--qenc_in_norm', type=int, default=0, help='Normalize the data before forwarding to the circuit')
  parser.add_argument('--qenc_out_norm', type=int, default=0, help='Normalize the output the circuit')
  
  parser.add_argument('--type_cost', type=str, default='rbf', help='Type of cost function in contrastive learning: diff,rbf')
  parser.add_argument('--type_anchor', type=int, default=1, help='Use anchor or not')

  parser.add_argument('--use_BP', type=int, default=0)
  parser.add_argument('--train_qfm', type=int, default=0, help='Training qfm or not')
  parser.add_argument('--layers', type=str, default='16,16,16')
  parser.add_argument('--n_wires', type=int, default=8, help='Number of wires') 
  parser.add_argument('--noise_level', type=int, default=0, help='Add noise to the data.')
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
  parser.add_argument('--dat_name', type=str, default='Fashion-Mnist', help='Name of the data')
  parser.add_argument('--n_test', type=int, default=10000, help='Number of test data')
  parser.add_argument('--n_train', type=int, default=1000, help='Number of train data')
  parser.add_argument('--n_labels', type=int, default=10, help='Number of labels')
  
  # For training
  parser.add_argument('--n_epochs_outer', type=int, default=200, help='Number of epoch outer')
  parser.add_argument('--n_epochs_inner', type=int, default=40, help='Number of epoch inner')
  
  # For system
  parser.add_argument('--rseed', type=int, default=0, help='Random seed')
  
  args = parser.parse_args()
  
  save_dir, nonlinear, use_BP, n_wires, n_obs = args.save_dir, args.nonlinear, args.use_BP, args.n_wires, args.n_obs
  # repeat encoding
  rep_enc, residual, type_cost, type_anchor, train_qfm = args.rep_enc, args.residual, args.type_cost, args.type_anchor, args.train_qfm
  weight_decay, lr, scale = args.weight_decay, args.lr, args.scale
  rc = args.rc
  dat_name, n_train, n_test, n_labels = args.dat_name, args.n_train, args.n_test, args.n_labels
  n_epochs_outer, n_epochs_inner = args.n_epochs_outer, args.n_epochs_inner
  rseed = args.rseed
  use_record, n_shots, n_basis = args.use_record, args.n_shots, args.n_basis
  qenc_in_norm, qenc_out_norm = args.qenc_in_norm, args.qenc_out_norm

  layers = [int(x) for x in args.layers.split(',')]
  lay_str = '_'.join([str(x) for x in layers])

  # Create folder to save results
  
  if 'qenc' in nonlinear:
    if n_basis > 1:
      nonlinear_str = f'{nonlinear}_basis_{n_basis}'
    else:
      nonlinear_str = f'{nonlinear}'
    nonlinear_str = f'{nonlinear_str}_BP_{use_BP}_obs_{n_obs}_shots_{n_shots}'
  else:
    nonlinear_str = f'{nonlinear}'
  
  basename = f'{nonlinear_str}_qfm_{train_qfm}_ionorm_{qenc_in_norm}_{qenc_out_norm}_res_{residual}_cost_{type_cost}_anchor_{type_anchor}_layers_{lay_str}_dat_{n_train}_{n_test}_epoch_{n_epochs_outer}_{n_epochs_inner}_scale_{scale}_lr_{lr}_rc_{rc}_rseed_{rseed}'

  torch.set_default_dtype(torch.double)
  random.seed(rseed)
  np.random.seed(rseed)
  torch.manual_seed(rseed)

  #train_loader, val_loader = Dat_FMnist(n_train, n_test, n_train, n_test, rseed=rseed)

  net = DeepNet(dims=layers, n_epochs_outer=n_epochs_outer, n_epochs_inner=n_epochs_inner, n_obs=n_obs, use_BP=use_BP,\
          qenc_in_norm=qenc_in_norm, qenc_out_norm=qenc_out_norm, n_basis=n_basis,
          rep_enc=rep_enc, residual=residual, train_qfm=train_qfm, type_cost=type_cost, type_anchor=type_anchor, n_wires=n_wires, weight_decay=weight_decay, lr=lr,\
          rc=rc, scale=scale, use_record=use_record, n_shots=n_shots, n_labels=n_labels, nonlinear=nonlinear)
  
  ed = EffectiveDimension(net, num_thetas=10, num_inputs=100, seed=rseed)
  f, trace = ed.get_fhat()
  # range of the number of data
  ndat_ls = [5000, 8000, 10000, 40000, 60000, 100000, 150000, 200000, 500000, 1000000]
  effdim = ed.eff_dim(f, ndat_ls)
  norm_effdim = np.array(effdim) / ed.d
  print('Effdim', effdim)
  print('Normalized effdim', norm_effdim)