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
from torch.utils.data import TensorDataset

import pickle
import os

from sklearn.model_selection import train_test_split

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

class CombinedTrainDatasetQPM(Dataset):
  def __init__(self, train_data_qpm, train_labels_qpm):
      self.train_data_qpm = train_data_qpm
      self.train_labels_qpm = train_labels_qpm
      
  def __getitem__(self, index):
      train_data_qpm_item = self.train_data_qpm[index]
      train_labels_qpm_item = self.train_labels_qpm[index]
      return train_data_qpm_item, train_labels_qpm_item
  
  def __len__(self):
      return len(self.train_data_qpm)

class CombinedTestDatasetQPM(Dataset):
  def __init__(self, test_data_qpm, test_labels_qpm):
      self.test_data_qpm = test_data_qpm
      self.test_labels_qpm = test_labels_qpm
      
  def __getitem__(self, index):
      test_data_qpm_item = self.test_data_qpm[index]
      test_labels_qpm_item = self.test_labels_qpm[index]
      return test_data_qpm_item, test_labels_qpm_item
  
  def __len__(self):
      return len(self.test_data_qpm)
  
# Function to filter by labels
def filter_by_label(dataset, labels):
  indices = []
  for i in range(len(dataset)):
      if dataset[i][1] in labels:  # dataset[i][1] represents the label
          indices.append(i)
  return indices  

def load_ground_states(file_path):
    """
    Reads a .pkl file and returns only the ground states from the loaded data.
    
    :param file_path: str - path to the .pkl file
    :return: list of ground states
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")
    
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    
    # Extract only the ground states from the loaded data
    ground_states = [entry["ground_state"] for entry in data]
    ground_states = np.array(ground_states, dtype=np.complex64)

    return ground_states

# load ground states of Ising cluster model
def read_IsingCluster_file(fname, n_qubits):
    data, labels = None, None
    if os.path.isfile(fname):
        data = load_ground_states(fname)

    labels_file = fname.replace('_data.pkl', '_labels.pkl')
    if os.path.isfile(labels_file):
        with open(labels_file, 'rb') as file:
            labels = pickle.load(file)
    return data, labels


# Another difficult IsingCluster_data  "gch" (Generalized Cluster Hamiltonian)
def read_gch_data(n_qubits, batch_size=0):
    # read / import data 
    training_fname = f"../data/phase_detection_gch_nq_{n_qubits}/ground_nq_{n_qubits}_train_50_gch_train_data.pkl"
    train_data, train_labels = read_IsingCluster_file(training_fname, n_qubits)

    test_fname = f"../data/phase_detection_gch_nq_{n_qubits}/ground_nq_{n_qubits}_test_1000_gch_test_data.pkl"
    test_data, test_labels = read_IsingCluster_file(test_fname, n_qubits)

    # combine data and label
    combined_train_dataset = CombinedTrainDatasetQPM(train_data, train_labels)
    combined_test_dataset = CombinedTestDatasetQPM(test_data, test_labels)
    
    # Shuffle data
    if batch_size == 0:
       train_batch_size = len(combined_train_dataset)
       test_batch_size = len(combined_test_dataset)
    else:
        train_batch_size = batch_size
        test_batch_size = batch_size
    
    combined_train_loader = DataLoader(combined_train_dataset, batch_size=train_batch_size, shuffle=True) # check
    combined_test_loader = DataLoader(combined_test_dataset, batch_size=test_batch_size, shuffle=False) # check
    
    return combined_train_loader, combined_test_loader

def read_dlp_data(dat_file, p, g, sls, rseed):
  # Load the state vectors from the .npz file
  if not os.path.isfile(dat_file):
      raise FileNotFoundError(f"The file '{dat_file}' does not exist.")
  loaded = np.load(dat_file)
  state_vectors = {int(key.split('_')[1]): loaded[key] for key in loaded.files}

  # Precompute g^i mod p and log_g
  g_powers = {i: pow(g, i, p) for i in range(p-1)}
  log_g = {v: k for k, v in g_powers.items()}
  yls = log_g.keys()

  labels = {}
  for y in yls:
     labels[y] = 0  # Initialize all labels to 0

  m = len(sls)
  for i, s in enumerate(sls):
    # Compute the interval [s, s + (p-3)/2] mod p
    upper = int(s + (p-3)/(2*m))
    if upper >= p:
        interval = set(range(s, p)) | set(range(0, (upper % p) + 1))
    else:
        interval = set(range(s, upper + 1))
    interval = interval & set(range(p-1))  # \log_g(y) in {0, ..., p-2}
    #print(f"Interval [s, s + (p-3)/2] mod p: {min(interval)} to {max(interval)}")

    # Assign labels f_s(y)
    
    for y in yls:
        log_y = log_g[y]
        lb = 1 if log_y in interval else 0
        labels[y] += lb

  for y in yls:
    if labels[y] > 0:
       labels[y] = 1
      
  # Prepare data
  data = [(y, state_vectors[y], labels[y]) for y in yls]

  # Split into training and testing sets (equal size)
  train_data, test_data = train_test_split(data, test_size=0.5, random_state=rseed, shuffle=True)

  # Separate into states and labels
  train_states = np.array([item[1] for item in train_data])
  train_labels = np.array([item[2] for item in train_data])
  train_y = np.array([item[0] for item in train_data])
  test_states = np.array([item[1] for item in test_data])
  test_labels = np.array([item[2] for item in test_data])
  test_y = np.array([item[0] for item in test_data])

  return train_states, train_labels, train_y, test_states, test_labels, test_y

def load_dlp_data(dat_file, p, g, sls, rseed, batch_size=0):
  train_states, train_labels, train_y, test_states, test_labels, test_y = read_dlp_data(dat_file, p, g, sls, rseed)
  # combine data and label
  combined_train_dataset = CombinedTrainDatasetQPM(train_states, train_labels)
  combined_test_dataset = CombinedTestDatasetQPM(test_states, test_labels)

  # Shuffle data
  if batch_size == 0:
      train_batch_size = len(combined_train_dataset)
      test_batch_size = len(combined_test_dataset)
  else:
      train_batch_size = batch_size
      test_batch_size = batch_size
  combined_train_loader = DataLoader(combined_train_dataset, batch_size=train_batch_size, shuffle=True)
  combined_test_loader = DataLoader(combined_test_dataset, batch_size=test_batch_size, shuffle=False)

  return combined_train_loader, combined_test_loader

def convert_labels(labels):
    """
    Convert labels according to the specified mapping:
    1 -> 0, 2 -> 1, 3 -> 2
    """
    label_mapping = {1: 0, 2: 1, 3: 2}
    return [label_mapping.get(label, label) for label in labels]

# Another difficult IsingCluster_data  "ssh" (Su-Schrieffer-Heeger)
def read_ssh_data(n_qubits, batch_size=0):
    # read / import data 
    training_fname = f"../data/phase_detection_ssh_nq_{n_qubits}/ground_nq_{n_qubits}_train_50_ssh_train_data.pkl"
    train_data, train_labels = read_IsingCluster_file(training_fname, n_qubits)

    test_fname = f"../data/phase_detection_ssh_nq_{n_qubits}/ground_nq_{n_qubits}_test_1000_ssh_test_data.pkl"
    test_data, test_labels = read_IsingCluster_file(test_fname, n_qubits)

    # Convert labels: 1 -> 0, 2 -> 1, 3 -> 2
    train_labels = convert_labels(train_labels)
    test_labels = convert_labels(test_labels)

    # combine data and label
    combined_train_dataset = CombinedTrainDatasetQPM(train_data, train_labels)
    combined_test_dataset = CombinedTestDatasetQPM(test_data, test_labels)
    
    # Shuffle data
    if batch_size == 0:
       train_batch_size = len(combined_train_dataset)
       test_batch_size = len(combined_test_dataset)
    else:
        train_batch_size = batch_size
        test_batch_size = batch_size
    combined_train_loader = DataLoader(combined_train_dataset, batch_size=train_batch_size, shuffle=True) # check
    combined_test_loader = DataLoader(combined_test_dataset, batch_size=test_batch_size, shuffle=False) # check

    return combined_train_loader, combined_test_loader

# Another version of Dat_QPM to create train_data and test_data
def read_IsingCluster_data(n_qubits, train_dat_type='para', test_dat_type='all', batch_size=0):
    # read / import data 
    training_fname = f"../data/phase_detection_python_periodic/ground_nq_{n_qubits}_train_40_data.pkl"
    train_data, train_labels = read_IsingCluster_file(training_fname, n_qubits)

    # Add train data from anti-ferro
    training_fname2 = f"../data/phase_detection_python_periodic/ground_nq_{n_qubits}_test_40_anti_ferro_{5}_data.pkl"
    train_data2, train_labels2 = read_IsingCluster_file(training_fname2, n_qubits)

    if train_dat_type == 'anti_ferro':
      train_data = train_data2
      train_labels = train_labels2
    elif train_dat_type == 'all':
      train_data = np.concatenate([train_data, train_data2])
      train_labels = np.concatenate([train_labels, train_labels2])

    test_data_list, test_labels_list = [], []
    if test_dat_type == 'all':
      test_dat_ls = ['para', 'anti_ferro']
    else:
      test_dat_ls =  [test_dat_type]

    for tmp in test_dat_ls:
      for k in range(10):
          test_fname = f"../data/phase_detection_python_periodic/ground_nq_{n_qubits}_test_40_{tmp}_{k}_data.pkl"
          test_data, test_labels = read_IsingCluster_file(test_fname, n_qubits)
          test_data_list.append(test_data)
          test_labels_list.append(test_labels)
    
    test_data = np.concatenate(test_data_list)
    test_labels = np.concatenate(test_labels_list)

    # combine data and label
    combined_train_dataset = CombinedTrainDatasetQPM(train_data, train_labels)
    combined_test_dataset = CombinedTestDatasetQPM(test_data, test_labels)
    
    # Shuffle data
    if batch_size == 0:
       train_batch_size = len(combined_train_dataset)
       test_batch_size = len(combined_test_dataset)
    else:
        train_batch_size = batch_size
        test_batch_size = batch_size
    combined_train_loader = DataLoader(combined_train_dataset, batch_size=train_batch_size, shuffle=True) # check
    combined_test_loader = DataLoader(combined_test_dataset, batch_size=test_batch_size, shuffle=False) # check

    return combined_train_loader, combined_test_loader

# open boundary conditions data (9qubit)
def read_IsingCluster_data_open(n_qubits, train_dat_type='para', test_dat_type='all', batch_size=0):
    # read / import data 
    training_fname = f"../data/phase_detection_python_open_{n_qubits}qubit/ground_nq_{n_qubits}_train_40_data.pkl"
    train_data, train_labels = read_IsingCluster_file(training_fname, n_qubits)

    # Add train data from anti-ferro
    training_fname2 = f"../data/phase_detection_python_open_{n_qubits}qubit/ground_nq_{n_qubits}_test_40_anti_ferro_{5}_data.pkl"
    train_data2, train_labels2 = read_IsingCluster_file(training_fname2, n_qubits)

    if train_dat_type == 'anti_ferro':
      train_data = train_data2
      train_labels = train_labels2
    elif train_dat_type == 'all':
      train_data = np.concatenate([train_data, train_data2])
      train_labels = np.concatenate([train_labels, train_labels2])

    test_data_list, test_labels_list = [], []
    if test_dat_type == 'all':
      test_dat_ls = ['para', 'anti_ferro']
    else:
      test_dat_ls =  [test_dat_type]

    for tmp in test_dat_ls:
      for k in range(10):
          test_fname = f"../data/phase_detection_python_open_{n_qubits}qubit/ground_nq_{n_qubits}_test_40_{tmp}_{k}_data.pkl"
          test_data, test_labels = read_IsingCluster_file(test_fname, n_qubits)
          test_data_list.append(test_data)
          test_labels_list.append(test_labels)
    
    test_data = np.concatenate(test_data_list)
    test_labels = np.concatenate(test_labels_list)

    # combine data and label
    combined_train_dataset = CombinedTrainDatasetQPM(train_data, train_labels)
    combined_test_dataset = CombinedTrainDatasetQPM(test_data, test_labels)
    
    # Shuffle data
    if batch_size == 0:
       train_batch_size = len(combined_train_dataset)
       test_batch_size = len(combined_test_dataset)
    else:
        train_batch_size = batch_size
        test_batch_size = batch_size
    combined_train_loader = DataLoader(combined_train_dataset, batch_size=train_batch_size, shuffle=True) # check
    # print("len(combined_train_dataset)", len(combined_train_dataset))
    combined_test_loader = DataLoader(combined_test_dataset, batch_size=test_batch_size, shuffle=False) # check

    return combined_train_loader, combined_test_loader

# Quantum-Phases-of-Matter Data
def Dat_QPM(num_qubits):
  ## read / import data 
  training_fname = "../data_fldr/dataset_n={}_train.txt".format(num_qubits)
  test_fname = "../data_fldr/dataset_n={}_test.txt".format(num_qubits)
  #test_fname = "../data_fldr/dataset_n={}_test_1280.txt".format(num_qubits)

  def read_eigenvectors(file):
    with open(file, 'r+') as f:
        textData = f.readlines()

        h_vals = []
        for i in range(len(textData)):
            h1h2, eigenvector = textData[i].split("_")

            h_vals.append(tuple(map(float, h1h2[1: -1].split(', '))))
            textData[i] = eigenvector

        return h_vals, np.loadtxt(textData, dtype=complex)
    
  h1h2_train_qpm, train_data_qpm = read_eigenvectors(training_fname)
  h1h2_test_qpm, test_data_qpm = read_eigenvectors(test_fname)

  # To get the correct labels of the training set we use the fact that data points with
  # h1 <= 1 are in the SPT phase and thus assigned the label 1 while h1 > 1 are in the paramagnetic phase 
  # This is only true for the training set which has h2 = 0 for all samples, 

  train_labels_qpm = np.zeros(40)
  # test_labels_qpm = np.zeros(192)
  test_labels_qpm = np.zeros(1280)

  for index, h1h2 in enumerate(h1h2_train_qpm):
      h1, h2 = h1h2
      if h1 <= 1: # check
          train_labels_qpm[index] = 1.0
  # https://github.com/Jaybsoni/Quantum-Convolutional-Neural-Networks/blob/main/qcnns_for_phase_recog/QCNNs%20for%20Classifying%20Quantum%20Phases%20of%20Matter.ipynb　に記載されているDMRGシミュレーションの結果を元にラベル付け
  for index, h1h2 in enumerate(h1h2_test_qpm):
      h1, h2 = h1h2
      # if h2 == 0.5033 and h1 < 0.4111:
      #     test_labels_qpm[index] = 1.0
      # if h2 == 0.3631 and h1 < 0.5667:
      #     test_labels_qpm[index] = 1.0
      # if h2 == 0.2229 and h1 < 0.7222:
      #     test_labels_qpm[index] = 1.0

      # para_mag_boundary
      if h2 == 0.8439 and h1 < 0.1000:
          test_labels_qpm[index] = 1.0
      if h2 == 0.6636 and h1 < 0.2556:
          test_labels_qpm[index] = 1.0  
      if h2 == 0.5033 and h1 < 0.4111:
          test_labels_qpm[index] = 1.0
      if h2 == 0.3631 and h1 < 0.5667:
          test_labels_qpm[index] = 1.0
      if h2 == 0.2229 and h1 < 0.7222:
          test_labels_qpm[index] = 1.0
      if h2 == 0.09766 and h1 < 0.8778:
          test_labels_qpm[index] = 1.0
      if h2 == -0.02755 and h1 < 1.0333:
          test_labels_qpm[index] = 1.0  
      if h2 == -0.1377 and h1 < 1.1889:
          test_labels_qpm[index] = 1.0
      if h2 == -0.2479 and h1 < 1.3444:
          test_labels_qpm[index] = 1.0
      if h2 == -0.3531 and h1 < 1.5000:
          test_labels_qpm[index] = 1.0

      # anti_ferro_mag_boundary
      if h2 == -1.004 and h1 > 0.1000:
          test_labels_qpm[index] = 1.0
      if h2 == -1.0009 and h1 > 0.2556:
          test_labels_qpm[index] = 1.0  
      if h2 == -1.024 and h1 > 0.4111:
          test_labels_qpm[index] = 1.0
      if h2 == -1.049 and h1 > 0.5667:
          test_labels_qpm[index] = 1.0
      if h2 == -1.079 and h1 > 0.7222:
          test_labels_qpm[index] = 1.0
      if h2 == -1.109 and h1 > 0.8778:
          test_labels_qpm[index] = 1.0
      if h2 == -1.154 and h1 > 1.0333:
          test_labels_qpm[index] = 1.0  
      if h2 == -1.225 and h1 > 1.1889:
          test_labels_qpm[index] = 1.0
      if h2 == -1.285 and h1 > 1.3444:
          test_labels_qpm[index] = 1.0
      if h2 == -1.35 and h1 > 1.5000:
          test_labels_qpm[index] = 1.0

  # combine data and label
  combined_train_dataset_qpm = CombinedTrainDatasetQPM(train_data_qpm, train_labels_qpm)
  combined_test_dataset_qpm = CombinedTestDatasetQPM(test_data_qpm, test_labels_qpm)
  
  # Shuffle data
  combined_train_loader_qpm = DataLoader(combined_train_dataset_qpm, batch_size=len(combined_train_dataset_qpm), shuffle=True) # check
  combined_test_loader_qpm = DataLoader(combined_test_dataset_qpm, batch_size=len(combined_test_dataset_qpm), shuffle=False) # check

  return combined_train_loader_qpm, combined_test_loader_qpm


def compute_entanglement_entropy(psi, n_subsystem):
    """
    Compute the entanglement entropy of a pure state |ψ⟩ for a subsystem.
    
    Args:
        psi (torch.Tensor): Pure state vector |ψ⟩ (batch_size, dim).
        n_subsystem (int): Number of qubits in subsystem A.
    
    Returns:
        torch.Tensor: Entanglement entropy for each state in the batch.
    """
    batch_size, dim = psi.shape
    n_total = int(torch.log2(torch.tensor(dim)))  # Total number of qubits
    n_remainder = n_total - n_subsystem          # Number of qubits in subsystem B
    
    if n_subsystem <= 0 or n_remainder <= 0:
        raise ValueError("Invalid subsystem size for partitioning.")
    
    # Reshape |ψ⟩ to a bipartite structure: (batch_size, 2^n_subsystem, 2^n_remainder)
    psi_reshaped = psi.view(batch_size, 2**n_subsystem, 2**n_remainder)
    
    # Compute reduced density matrix ρ_A = Tr_B(|ψ⟩⟨ψ|)
    rho_A = torch.einsum('bij,bkj->bik', psi_reshaped, psi_reshaped.conj())
    
    # Compute eigenvalues of ρ_A
    eigvals = torch.linalg.eigvalsh(rho_A)  # Shape: (batch_size, 2^n_subsystem)
    
    # Ensure non-negative eigenvalues (numerical stability)
    eigvals = torch.clamp(eigvals, min=1e-12)
    
    # Compute von Neumann entropy S(ρ_A) = -Tr(ρ_A log ρ_A)
    entropy = -torch.sum(eigvals * torch.log(eigvals), dim=1)
    return entropy


def preprocess_entanglement_entropy(data_loader, n_subsystem, batch_size, shuffle=True):
    """
    Create a DataLoader with quantum states as data and entanglement entropy as targets.
    
    Args:
        data_loader: Original DataLoader with pure states (|ψ⟩).
        n_subsystem: Number of qubits in subsystem A.
        batch_size: Desired batch size.
    
    Returns:
        DataLoader: DataLoader with data=|ψ⟩ and targets=entanglement entropy.
    """
    state_vectors = []
    entanglement_entropies = []

    for batch_states, _ in data_loader:  # Ignore labels from the original DataLoader
        psi = batch_states  # Shape: (batch_size, dim)
        
        # Compute entanglement entropy
        entropy = compute_entanglement_entropy(psi, n_subsystem)
        
        state_vectors.append(psi)
        entanglement_entropies.append(entropy)

    # Combine all batches
    state_vectors = torch.cat(state_vectors, dim=0)
    entanglement_entropies = torch.cat(entanglement_entropies, dim=0)

    # Create TensorDataset and DataLoader
    dataset = TensorDataset(state_vectors, entanglement_entropies)
    new_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return new_dataloader


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

def add_X_rot_noise_data(x, n_wires, mag=0.1, bg=0, interval=1):
  qdev = tq.QuantumDevice(n_wires=n_wires, bsz=x.shape[0], device=x.device)
  qdev.set_states(x) 
  qdev.states = qdev.states.to(torch.complex64)

  random_angles = np.random.uniform(0, 2 * np.pi, size=n_wires) * mag
  for j in range(bg, n_wires, interval):
    tqf.rx(qdev, wires=j, params=random_angles[j])
  return qdev.states.reshape(x.shape)

import torch

def add_X_rot_noise_data_loader(data_loader, n_wires, noise_level):
    """
    Convert a DataLoader into a noisy version by applying X-rotation noise to the data.

    Args:
        data_loader (torch.utils.data.DataLoader): Original data loader containing clean data.
        n_wires (int): Number of qubits (wires) in the quantum system.
        noise_level (float): Magnitude of X-rotation noise to apply.

    Returns:
        torch.utils.data.DataLoader: Noisy version of the input data loader.
    """
    # Lists to store noisy data and corresponding labels
    noisy_data = []
    noisy_labels = []

    for data, labels in data_loader:
        # Apply X-rotation noise to the quantum data
        noisy_batch = add_X_rot_noise_data(data, n_wires=n_wires, mag=noise_level)
        noisy_data.append(noisy_batch)
        noisy_labels.append(labels)

    # Combine all noisy data and labels into tensors
    noisy_data = torch.cat(noisy_data)
    noisy_labels = torch.cat(noisy_labels)

    # Create a new dataset and data loader
    noisy_dataset = torch.utils.data.TensorDataset(noisy_data, noisy_labels)
    noisy_loader = torch.utils.data.DataLoader(noisy_dataset, batch_size=data_loader.batch_size, shuffle=False)

    return noisy_loader

def add_label_flip_data_loader(data_loader, p_flip):
    """
    Convert a DataLoader into a noisy version by applying label flip to the data.

    Args:
        data_loader (torch.utils.data.DataLoader): Original data loader containing clean data.
        p_flip (float): Proportion of labels to flip.

    Returns:
        torch.utils.data.DataLoader: Noisy version of the input data loader.
    """
    # Lists to store noisy data and corresponding labels
    noisy_data = []
    noisy_labels = []

    for data, labels in data_loader:
        # Apply label noise to the labels
        num_labels_to_flip = int(len(labels) * p_flip)
        indices_to_flip = np.random.choice(len(labels), size=num_labels_to_flip, replace=False)
        noisy_labels_batch = labels.clone()
        noisy_labels_batch[indices_to_flip] = 1 - noisy_labels_batch[indices_to_flip]

        noisy_data.append(data)
        noisy_labels.append(noisy_labels_batch)

    # Combine all noisy data and labels into tensors
    noisy_data = torch.cat(noisy_data)
    noisy_labels = torch.cat(noisy_labels)

    # Create a new dataset and data loader
    noisy_dataset = torch.utils.data.TensorDataset(noisy_data, noisy_labels)
    noisy_loader = torch.utils.data.DataLoader(noisy_dataset, batch_size=data_loader.batch_size, shuffle=False)

    return noisy_loader
