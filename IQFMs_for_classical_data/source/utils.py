import torch
import numpy as np
#import torchvision
#from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader, Subset, Dataset, ConcatDataset
#from torch.utils.data.dataloader import DataLoader
#from torchvision.datasets import CIFAR10

import torchvision.transforms as transforms
from torchvision import datasets
#from torchvision import transforms

# For torch quantum
import torchquantum as tq
import torchquantum.functional as tqf

RELU = 'relu'
GELU = 'gelu'
SGM = 'sigmoid'
TANH = 'tanh'
ID = 'id'

class CombinedTrainDataset(Dataset):
  def __init__(self, train_dataset, train_dataset_rot):
      self.train_dataset = train_dataset
      self.train_dataset_rot = train_dataset_rot
      
  def __getitem__(self, index):
      train_dataset_item = self.train_dataset[index]
      train_dataset_rot_item = self.train_dataset_rot[index]
      return train_dataset_item, train_dataset_rot_item
  
  def __len__(self):
      return len(self.train_dataset)

class CombinedValDataset(Dataset):
  def __init__(self, val_dataset, val_dataset_rot):
      self.val_dataset = val_dataset
      self.val_dataset_rot = val_dataset_rot
      
  def __getitem__(self, index):
      val_dataset_item = self.val_dataset[index]
      val_dataset_rot_item = self.val_dataset_rot[index]
      return val_dataset_item, val_dataset_rot_item
  
  def __len__(self):
      return len(self.val_dataset)
  
# Function to filter by labels
def filter_by_label(dataset, labels):
  indices = []
  for i in range(len(dataset)):
      if dataset[i][1] in labels:  # dataset[i][1] represents the label
          indices.append(i)
  return indices  

# Function to Add Gaussian Noise
def add_gaussian_noise(tensor, mean=0.0, stddev=0.0):
    noise = torch.randn(tensor.size()) * stddev + mean
    noisy_tensor = tensor + noise
    return noisy_tensor

def sample_equal_per_class(dataset, labels, n_samples):
  n_labels = len(labels)
  n_samples_per_class = n_samples // n_labels
  remainder = n_samples % n_labels

  indices = []
  for label in labels:
    label_indices = [i for i, (_, lbl) in enumerate(dataset) if lbl == label]
    sampled_indices = np.random.choice(label_indices, n_samples_per_class, replace=False)
    indices.extend(sampled_indices)

  if remainder > 0:
    remaining_indices = []
    for label in labels:
      label_indices = [i for i, (_, lbl) in enumerate(dataset) if lbl == label]
      remaining_indices.extend(np.random.choice(label_indices, remainder, replace=False))
    np.random.shuffle(remaining_indices)
    indices.extend(remaining_indices[:remainder])
  return indices

# Fashion-MNIST data
def Dat_FMnist(batch_size_train, batch_size_test, n_train=None, n_test=None, rseed=0, labels=None, noise_level=0):
  compress_factor = 1
  reshape_f = lambda x: torch.reshape(x[0, ::compress_factor, ::compress_factor], (-1, ))
  np.random.seed(rseed)
  torch.manual_seed(rseed)
  # print("noise_level", noise_level)
  transforms_train = transforms.Compose([
    #transforms.RandomHorizontalFlip(), # FLips the image w.r.t horizontal axis
    #transforms.RandomRotation(5),     #Rotates the image to a specified angel
    transforms.RandomAffine(0, translate= (.025,.025),shear=0), #Performs actions like zooms, change shear angles.
    #transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1), # Set the color params
    transforms.ToTensor(),
    transforms.Lambda(lambda x: add_gaussian_noise(x, stddev=noise_level)),
    transforms.Normalize((0.1307,), (0.3081,)),
    transforms.Lambda(lambda x: torch.flatten(x)) 
    ])
  transforms_val = transforms.Compose([transforms.ToTensor(),  transforms.Normalize((0.1307,), (0.3081,)),transforms.Lambda(lambda x: torch.flatten(x))])
  
  # Transformations for data with 90-degree left rotation
  transforms_train_rot = transforms.Compose([
      transforms.RandomRotation(90),
      transforms.RandomAffine(0, translate=(.025, .025), shear=0),
      transforms.ToTensor(),
      transforms.Lambda(lambda x: add_gaussian_noise(x, stddev=noise_level)),
      transforms.Normalize((0.1307,), (0.3081,)),
      transforms.Lambda(lambda x: torch.flatten(x))
  ])
  
  transforms_val_rot = transforms.Compose([
      transforms.RandomRotation(90),
      transforms.ToTensor(),
      transforms.Normalize((0.1307,), (0.3081,)),
      transforms.Lambda(lambda x: torch.flatten(x))
  ])

  train_dataset = datasets.FashionMNIST("~/.pytorch/F_MNIST_data/", train=True, download=True, transform=transforms_train)
  val_dataset = datasets.FashionMNIST("~/.pytorch/F_MNIST_data/", train=False, download=True, transform=transforms_val)

  train_dataset_rot = datasets.FashionMNIST("~/.pytorch/F_MNIST_data/", train=True, download=True, transform=transforms_train_rot)
  val_dataset_rot = datasets.FashionMNIST("~/.pytorch/F_MNIST_data/", train=False, download=True, transform=transforms_val_rot)

  if labels is not None:
    # Filtering both training data and test data
    train_ids = filter_by_label(train_dataset, labels)
    val_ids = filter_by_label(val_dataset, labels)

    train_ids_rot = filter_by_label(train_dataset_rot, labels)
    val_ids_rot = filter_by_label(val_dataset_rot, labels)   

    # Create Subset based on the filtering result
    train_dataset = Subset(train_dataset, train_ids)
    val_dataset = Subset(val_dataset, val_ids)

    train_dataset_rot = Subset(train_dataset_rot, train_ids_rot)
    val_dataset_rot = Subset(val_dataset_rot, val_ids_rot)

    combined_train_dataset = CombinedTrainDataset(train_dataset, train_dataset_rot)
    combined_val_dataset = CombinedValDataset(val_dataset, val_dataset_rot)

  if n_train is not None:
    sub_ids = sample_equal_per_class(train_dataset, labels, n_train)
    # sub_ids = np.random.choice(len(train_dataset), n_train, replace=False)
    #print('n_train_sub_ids', sub_ids)
    combined_train_loader = DataLoader(Subset(combined_train_dataset, sub_ids), batch_size=batch_size_train, shuffle=True)
  else:
    combined_train_loader = DataLoader(combined_train_dataset, batch_size=batch_size_train, shuffle=True)

  if n_test is not None:
    sub_ids = sample_equal_per_class(val_dataset, labels, n_test)
    # sub_ids = np.random.choice(len(val_dataset), n_test, replace=False)
    #print('n_test_sub_ids', sub_ids)
    combined_val_loader = DataLoader(Subset(combined_val_dataset, sub_ids), batch_size=batch_size_test, shuffle=False)
  else:
    combined_val_loader = DataLoader(combined_val_dataset, batch_size=batch_size_test, shuffle=False)

  return combined_train_loader, combined_val_loader

def overlay_data_with_labels(x, y, n_labels):
    
    x_ext = x.clone()
    # only column y-th is set to a scalar value x.max() 
    # other columns less than n_labels are set to zero
    x_ext[:, :n_labels] *= 0.0
    x_ext[range(x.shape[0]), y] = x.max()

    return x_ext

def gen_encoder(n_wires, nonlinear):
    if 'qenc_44rxyzx' in nonlinear:
      enc_list = []
      input_idx = 0
      for func in ['rx', 'ry', 'rz', 'rx']:
        for w in range(n_wires):
          enc_list.append({"input_idx": [input_idx], "func": func, "wires": [w]})
          input_idx += 1
      encoder = tq.GeneralEncoder(enc_list)
    elif 'qenc_44hrxyzx' in nonlinear:
      enc_list = []
      for w in range(n_wires):
        enc_list.append({"func": "h", "wires": [w]})
      input_idx = 0
      for func in ['rx', 'ry', 'rz', 'rx']:
        for w in range(n_wires):
          enc_list.append({"input_idx": [input_idx], "func": func, "wires": [w]})
          input_idx += 1
      encoder = tq.GeneralEncoder(enc_list)
    elif 'qenc_44rxxyyzzxx' in nonlinear:
      enc_list = []
      input_idx = 0
      for func in ['rxx', 'ryy', 'rzz', 'rxx']:
        for w in range(n_wires):
          w_next = (w+1) % n_wires
          enc_list.append({"input_idx": [input_idx], "func": func, "wires": [w, w_next]})
          input_idx += 1
      encoder = tq.GeneralEncoder(enc_list)
    elif 'qenc_44rycx' in nonlinear:
      enc_list = []
      input_idx = 0
      for _ in range(4):
        for w in range(n_wires):
          enc_list.append({"input_idx": [input_idx], "func": "ry", "wires": [w]})
          input_idx += 1
        for w in range(n_wires - 1):
          # Linear entanglement
          enc_list.append({"input_idx": None, "func": "cx", "wires": [w, w+1]})
      encoder = tq.GeneralEncoder(enc_list)
    elif 'qenc_44rzcx' in nonlinear: #Not good for initial state 0
      enc_list = []
      input_idx = 0
      for _ in range(4):
        for w in range(n_wires):
          enc_list.append({"input_idx": [input_idx], "func": "rz", "wires": [w]})
          input_idx += 1
        for w in range(n_wires - 1):
          # Linear entanglement
          enc_list.append({"input_idx": None, "func": "cx", "wires": [w, w+1]})
      encoder = tq.GeneralEncoder(enc_list)  
    elif 'qenc_44ryzcx' in nonlinear: #Not good for initial state 0
      enc_list = []
      input_idx = 0
      for _ in range(2):
        for w in range(n_wires):
          for func in ['ry', 'rz']:
            enc_list.append({"input_idx": [input_idx], "func": func, "wires": [w]})
            input_idx += 1
        for w in range(n_wires - 1):
          # Linear entanglement
          enc_list.append({"input_idx": None, "func": "cx", "wires": [w, w+1]})
      encoder = tq.GeneralEncoder(enc_list)  
    elif 'qenc_44u3rx' in nonlinear:
      encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_u3rx"])
    elif 'qenc_44u3hrx' in nonlinear:
      encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_u3_h_rx"])
    elif 'qenc_44ryzxy' in nonlinear:
      encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
    elif 'qenc_44rzsx' in nonlinear:
      encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_rzsx"])
    elif 'qenc_44rysx' in nonlinear:
      enc_list = []
      input_idx = 0
      for _ in range(4):
        for w in range(n_wires):
          enc_list.append({"input_idx": [input_idx], "func": "ry", "wires": [w]})
          input_idx += 1
        for w in range(n_wires):
          enc_list.append({"input_idx": None, "func": "sx", "wires": [w]})
      encoder = tq.GeneralEncoder(enc_list)  
    elif 'qenc_44ryzsx' in nonlinear:
      enc_list = []
      input_idx = 0
      for _ in range(2):
        for w in range(n_wires):
          for func in ['ry', 'rz']:
            enc_list.append({"input_idx": [input_idx], "func": func, "wires": [w]})
            input_idx += 1
        for w in range(n_wires):
          enc_list.append({"input_idx": None, "func": "sx", "wires": [w]})
      encoder = tq.GeneralEncoder(enc_list)
    else:
      encoder = tq.GeneralEncoder()
    idx_ls = []
    for func in encoder.func_list:
      if 'input_idx' in func.keys() and func['input_idx'] is not None:
        idx_ls.extend(func['input_idx'])
    return encoder, len(idx_ls)