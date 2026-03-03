import matplotlib.pyplot as plt
import seaborn as sns
import os
import random

import torch
import torch.nn as nn

def qenc_out_normalize(h, qenc_out_norm):
  if qenc_out_norm == 1:
    h = torch.nn.Sigmoid()(h) * (2*torch.pi) - torch.pi
  elif qenc_out_norm == 2:
    h = torch.nn.Tanh()(h) * (torch.pi)
  elif qenc_out_norm == 3:
    h = torch.nn.GELU()(h)
  return h

def normalize(x):
  return x / (x.norm(2, 1, keepdim=True) + 1e-5)

def sim_func(h, pivot):
  cos = nn.CosineSimilarity(dim=1, eps=1e-6)
  return cos(h, pivot)
  #return torch.sum(h, 1)

def over_data_with_tensor_product(state, label, num_labels):
  # Adjust num_labels to be even if necessary
  if num_labels % 2 != 0:
      num_labels += 1  # Make it even by adding one
  # Represent label as a one-hot encoded quantum state
  label_tensor = torch.zeros(num_labels, dtype=state.dtype, device=state.device)
  # print(f"Label: {label}, Label Tensor Size: {num_labels}")  
  # print(f"Num Labels: {num_labels}")  
  label_tensor[label] = 1.0  # One-hot encoding
  return torch.kron(state, label_tensor)

def extend_data_with_tensor_product(data, labels, num_labels):
    """
    Extend the dataset by taking the tensor product of each data point and its true label.
    
    Args:
        data (torch.Tensor): Dataset of quantum states (N x d), where N is the number of samples.
        labels (torch.Tensor): Labels corresponding to the data (N).
        num_labels (int): Total number of unique labels.
    
    Returns:
        torch.Tensor: Extended dataset with the tensor product of data and labels (N x (d * num_labels)).
    """
    extended_data = []

    for i in range(data.size(0)):
        current_state = data[i]
        current_label = labels[i].item()
        extended_state = over_data_with_tensor_product(current_state, current_label, num_labels)
        
        # Append to the extended dataset
        extended_data.append(extended_state)

    # Stack the extended dataset into a single tensor
    return torch.stack(extended_data)


def make_pos_neg_samples(train_data_qpm, train_labels_qpm, type, num_labels):
  if type == 'rand':
    return make_pos_neg_rand_sampling(train_data_qpm, train_labels_qpm)
  elif type == 'tensor':
    return make_pos_neg_tensor_product(train_data_qpm, train_labels_qpm, num_labels)
  else:
    raise ValueError(f"Invalid type: {type}")

def make_pos_neg_rand_sampling(train_data_qpm, train_labels_qpm):
  # two different view of data
  # train_data_qpm_noise1 = add_X_rot_noise_data(train_data_qpm, n_wires=self.n_wires, mag=1.0, bg=0, interval=2)
  # train_data_qpm_noise2 = add_X_rot_noise_data(train_data_qpm, n_wires=self.n_wires, mag=1.0, bg=1, interval=2)

  unique_labels = train_labels_qpm.unique().tolist()

  # Save the indices of each label
  label_to_indices = {label: (train_labels_qpm == label).nonzero(as_tuple=True)[0].tolist() for label in unique_labels}
  
  x_pos_list = []
  for i in range(train_data_qpm.size(0)):
    current_label = train_labels_qpm[i].item()
    # If the data is not SPT, we randomly select the positive sample having the same label
    if True:
      same_label_indices = label_to_indices[current_label]
      random_index = random.choice(same_label_indices)
      x_pos_list.append(train_data_qpm[random_index])
    else:
      # If the data is SPT, we randomly add X-rotation noise to the data to create positive sample
      x_pos_list.append(train_data_qpm_noise1[i])

  x_pos = torch.stack(x_pos_list)

  # Create a list of negative sample
  x_neg_list = []

  # Randomly select label that is not the same as the current label
  for i in range(train_data_qpm.size(0)):
      current_label = train_labels_qpm[i].item()
      other_indices = [index for label, indices in label_to_indices.items() if label != current_label for index in indices]

      if not other_indices:
          print(f"No other indices found for label {current_label}")
      # If the data is  SPT, we randomly select the sample having the different label as positive sample
      if True:
        random_index = random.choice(other_indices)
        x_neg_list.append(train_data_qpm[random_index])
      else:
        # If the data is not SPT, we add noise to destroy the data to create negative sample
        x_neg_list.append(train_data_qpm_noise2[i])

  x_neg = torch.stack(x_neg_list)
  return x_pos, x_neg
       
def make_pos_neg_tensor_product(train_data_qpm, train_labels_qpm, num_labels):
    """
    Create positive and negative samples for contrastive learning.
    - Positive sample: Tensor product of quantum state with its true label.
    - Negative sample: Tensor product of quantum state with a randomly chosen wrong label.
    
    Args:
        train_data_qpm (torch.Tensor): Quantum state samples (N x d).
        train_labels_qpm (torch.Tensor): Labels corresponding to the samples (N).
    
    Returns:
        torch.Tensor, torch.Tensor: Positive samples, Negative samples.
    """
    unique_labels = train_labels_qpm.unique().tolist()

    x_pos_list = []
    x_neg_list = []

    for i in range(train_data_qpm.size(0)):
        current_label = train_labels_qpm[i].item()
        current_state = train_data_qpm[i]
        
        # Positive sample: Tensor product with true label
        combined_x = over_data_with_tensor_product(current_state, current_label, num_labels)
        x_pos_list.append(combined_x) 

        # Negative sample: Tensor product with a randomly chosen wrong label
        wrong_labels = [label for label in unique_labels if label != current_label]
        random_wrong_label = random.choice(wrong_labels)

        wrong_label_state = over_data_with_tensor_product(current_state, random_wrong_label, num_labels)
        x_neg_list.append(wrong_label_state)  # Tensor product

    # Stack the positive and negative samples
    x_pos = torch.stack(x_pos_list)
    x_neg = torch.stack(x_neg_list)
    
    return x_pos, x_neg

   
def get_num_out_features(linear_out_fes, nonlinear, n_wires, n_obs, n_shots, use_record, n_fet_enc, n_basis=1):
  if 'qenc' in nonlinear:
    # Number of circuit
    n_c = int(linear_out_fes / n_fet_enc)

    if use_record == 3: 
      # To increase measurement information, add observables.  
      # For 8 qubits, the total number of observables is 312: {𝛼_𝑖}, {𝛼_𝑖 𝛽_(𝑖+1)}, {𝛼_𝑖 𝛽_(𝑖+1) 𝛾_(𝑖+2)}, where 𝛼, 𝛽, 𝛾 ∈ {𝑋, 𝑌, 𝑍}.      n_fet_out = 312
      # n_c = 1 # TODO:Temporary code. revise later. Set n_c to 1 so that num_nonlinear_out becomes 312.
      n_fet_out = n_wires * (3 + 3*3 + 3*3*3)
      # print("use_record == 3")
    elif use_record == 1  and n_shots > 0:
      # if we use measurement records
      n_fet_out = n_wires * n_shots * n_basis
    elif n_obs > 0:
      # if we specify the number of observations and number of bases
      n_fet_out = n_obs * n_basis
    elif n_obs == 0 and use_record == 2:
      n_fet_out = (2**n_wires - 1) * n_basis
    else:
      # just measure all qubits
      n_fet_out = n_wires * n_basis      
    num_nonlinear_out = n_fet_out * n_c
    return num_nonlinear_out
  else:
    return linear_out_fes

def count_model_params(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)

def visualize_linear_layer(layer, basename, epoch=None, track_weights=None, save_dir=None, ftypes=['png']):
    """
    Visualize and optionally save the weights and biases of an nn.Linear layer.

    Args:
        layer (torch.nn.Linear): The linear layer to visualize.
        epoch (int, optional): Current epoch (used for tracking weight evolution). Default: None.
        track_weights (list, optional): List to store weight evolution across epochs. Default: None.
        save_dir (str, optional): Directory to save the visualizations as images. Default: None.

    Returns:
        None
    """
    if not hasattr(layer, "P"):
        raise ValueError("The provided layer does not have a 'P' attribute. "
                         "Please ensure it is a torch.nn.Linear layer.")
        return

    # Ensure save directory exists
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Extract weights and biases
    weights = layer.P.weight.detach().cpu().numpy()
    biases = layer.P.bias.detach().cpu().numpy()

    if epoch is not None:
       basename = f"{basename}_epoch_{epoch}"

    # Plot weight and bias distributions
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.hist(weights.flatten(), bins=30, color='blue', alpha=0.7)
    plt.title(f"Weight Distribution {basename}")
    plt.xlabel("Weight Value")
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    plt.hist(biases.flatten(), bins=30, color='green', alpha=0.7)
    plt.title(f"Bias Distribution {basename}")
    plt.xlabel("Bias Value")
    plt.ylabel("Frequency")

    plt.tight_layout()
    if save_dir:
        file_name = f"{basename}_weights_biases" if epoch is not None else f"{basename}_weights_biases"
        for ftype in ftypes:
            plt.savefig(os.path.join(save_dir, f"{file_name}.{ftype}"))
    plt.show()

    # Visualize weight matrix as a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(weights, cmap="coolwarm", center=0, cbar=True)
    plt.title(f"Weight Matrix Heatmap {basename}")
    plt.xlabel("Input Features")
    plt.ylabel("Output Features")

    if save_dir:
        file_name = f"{basename}_heatmap" if epoch is not None else f"{basename}_heatmap"
        for ftype in ftypes:
          plt.savefig(os.path.join(save_dir, f"{file_name}.{ftype}"))
    plt.show()

    # Track weight evolution if enabled
    if track_weights is not None and epoch is not None:
        track_weights.append(weights.flatten())
        plt.figure(figsize=(10, 6))
        for i, tracked_weights in enumerate(track_weights):
            plt.hist(tracked_weights, bins=30, alpha=0.5, label=f"Epoch {i+1}")
        plt.legend()
        plt.title(f"Weight Distribution Evolution {basename}")
        plt.xlabel("Weight Value")
        plt.ylabel("Frequency")

        if save_dir:
            file_name = f"{basename}_weight_evolution"
            for ftype in ftypes:
              plt.savefig(os.path.join(save_dir, f"{file_name}.{ftype}"))
        plt.show()
    plt.close()

