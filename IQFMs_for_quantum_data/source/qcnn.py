import torch
import torch.nn as nn
import torch.nn.functional as F
# For torch quantum
import torchquantum as tq
import torchquantum.functional as tqf
from torchquantum.measurement import expval_joint_analytical, expval_joint_sampling_grouping

from utils import *

def parameter_shift(qdev, model, batch_data, batch_labels, loss_lb, scale):
    """
        Compute ∂L/∂θ for every θ in model.encoder.params using
        (i) one backward pass through the classical head to get dL/dy,
        (ii) two quantum executions per θ to get dy/dθ, then chain-rule.
    """
    B = batch_data.size(0)  # Batch size
    shift = np.pi / 2  # Shift value for parameter shift rule

    # --- 1) Get pre-FC outputs z0 and classical gradient dL/dz ---
    qdev.reset_states(bsz=B)
    with torch.no_grad():
        z0 = model.quantum_forward(batch_data, qdev)
    # now z0: shape [B]
    # make it require_grad so we can backprop through fc+loss
    z0 = z0.clone().detach().requires_grad_(True).unsqueeze(-1)  # [B,1]
    if model.n_labels > 2:
        y0 = model.fc(z0)                  # [B, C]
    else:
        # for binary or regression you might skip the FC
        y0 = z0                            # [B,1] or [B,]
    
    # compute the loss L(z0) = batch_loss(targets, y0)
    L0 = batch_loss(target=batch_labels, predict=y0,
                    n_labels=model.n_labels,
                    loss_lb=loss_lb,
                    scale=scale)
    # backprop *only* to z0
    (dL_dz0,) = torch.autograd.grad(L0, z0)  # shape [B,1]
    dL_dz0 = dL_dz0.view(B)                  # [B]
    
    # --- 2) Parameter-shift on the quantum part ---
    gradients = []
    for param in model.encoder.params:
        # print("param.numel()", param.numel())
        flat = param.data.view(-1)

        # Initialize gradient tensor for the current parameter
        param_gradients = torch.zeros_like(param)

        # Iterate over each element in the parameter tensor
        for idx in range(flat.numel()):
            # Store original parameter value
            original_value = flat[idx].item()
            
            # Compute forward shift
            with torch.no_grad():
                flat[idx] = original_value + shift
            qdev.reset_states(bsz=B)
            z_plus = model.quantum_forward(batch_data, qdev)
            
            # Compute backward shift
            with torch.no_grad():
                flat[idx] = original_value - shift
            qdev.reset_states(bsz=B)
            z_minus = model.quantum_forward(batch_data, qdev)
            
            # Reset parameter to original value
            with torch.no_grad():
                flat[idx] = original_value
            
            # Compute gradient using parameter shift rule
            dz = (z_plus - z_minus) / 2        # [B]
            param_gradients[idx] = torch.dot(dL_dz0, dz)  # dot product with dL/dz0
        
        # Append the gradient tensor for the current parameter
        gradients.append(param_gradients.view_as(param))   
    return gradients

class QCNNEncoder(tq.QuantumModule):
    def __init__(self, n_wires, var_depth):
        super().__init__()
        self.n_wires = n_wires
        self.var_depth = var_depth
        # self.n_layers = np.log2(n_wires).astype(int)  # Number of layers in the QCNN
        # self.n_layers = int(np.ceil(np.log2(n_wires)))
        self.n_layers = self.calculate_layers(n_wires)
        # print(f"n_layers {self.n_layers}")

        # Trainable parameters for RX, RZ, RX, and RZZ gates for each layer and depth
        self.params = nn.ParameterList([
            nn.Parameter(torch.rand(4 * self.calculate_active_wires(lay) * var_depth) * 2 * torch.pi)
            for lay in range(self.n_layers)
        ])

        for lay in range(self.n_layers):
            n_active_wires = self.calculate_active_wires(lay)
            # print(f"Layer {lay}: Active wires = {n_active_wires}, Params size = {len(self.params[lay])}")  

    def calculate_layers(self, n_wires):
        """
        Calculate the number of layers needed to reduce the wires to 1.
        """
        layers = 0
        while n_wires > 1:
            n_wires = (n_wires + 1) // 2  # Reduce the number of wires by half, rounding up
            layers += 1
        return layers

    def calculate_active_wires(self, layer):
        """
        Calculate the number of active wires for a given layer.
        """
        return int(np.ceil(self.n_wires / (2 ** layer)))      


    def apply_convolution(self, qdev, layer, active_wires):
        """
        Apply RX, RZ, RX gates and then RZZ gates for var_depth times in the given layer.
        """
        n_active = len(active_wires)
        # print(f"Layer {layer}, n_active = {n_active}")
        for d in range(self.var_depth):  # Repeat for var_depth
            for j in range(n_active):
                # Index for parameters
                param_index = 4 * j + d * 4 * n_active
                # print(f"Layer {layer}, Depth {d}, Index {j}: Param index = {param_index}, Param size = {len(self.params[layer])}")


                # RX, RZ, RX gates with trainable parameters
                qdev.rx(wires=active_wires[j], params=self.params[layer][param_index + 0])
                qdev.rz(wires=active_wires[j], params=self.params[layer][param_index + 1])
                qdev.rx(wires=active_wires[j], params=self.params[layer][param_index + 2])

            for bg in [0, 1]:
              for j in range(bg, n_active, 2):
                  param_index = 4 * j + d * 4 * n_active
                  jnext = (j + 1) % n_active  # Neighboring qubit for RZZ

                  # RZZ gate with trainable parameter
                  if jnext < n_active:  # Ensure within bounds
                      qdev.rzz(wires=[active_wires[j], active_wires[jnext]], 
                              params=self.params[layer][param_index + 3])


    def apply_pooling(self, qdev, active_wires):
        """
        Pooling without directly discarding qubits.
        Reduce qubits by half using CNOT gates and logical tracking.
        """
        if True:
            for j in range(0, len(active_wires), 2):  # Process pairs of qubits
                jnext = j + 1
                if jnext < len(active_wires):  # Ensure within bounds
                    qdev.cnot(wires=[active_wires[j], active_wires[jnext]])  # Entangle qubits
        # Keep only the first qubit in each pair
        active_wires = active_wires[::2]
        # print(f"len(active_wires) = {len(active_wires)}")
        # print("active_wires", active_wires)
        return active_wires

    def forward(self, qdev):
        """
        Forward pass through the QCNN encoder.
        """
        active_wires = list(range(self.n_wires))  # Start with all qubits active
        for lay in range(self.n_layers):  # Continue until only one qubit is active
            self.apply_convolution(qdev, layer=lay, active_wires=active_wires)
            if len(active_wires) > 1:
              active_wires = self.apply_pooling(qdev, active_wires)

            #print(f'lay={lay}, active qubits = {active_wires}')

        return active_wires


class QuantumConvolutionalModel(nn.Module):
    def __init__(self, n_wires, var_depth, n_labels, n_shots):
        super().__init__()
        self.encoder = QCNNEncoder(n_wires=n_wires, var_depth=var_depth)
        self.n_labels = n_labels
        self.n_shots = n_shots
        self.fc = None
        if n_labels > 2:
            self.fc = nn.Linear(1, n_labels) # Fully connected layer for multi-class classification

    def quantum_forward(self, x, qdev):
        """
        Quantum forward pass: Encode the input quantum state.
        and perform the measurement.
        """
        qdev.set_states(x)
        active_wires = self.encoder(qdev)  # Apply QCNN encoder
        
        # Measure the expectation value of the Pauli-Z operator on the final active qubit
        obs = ['I'] * self.encoder.n_wires
        obs[active_wires[0]] = 'Z'
        observable = ''.join(obs)
        if self.n_shots == 0:
            output = expval_joint_analytical(qdev, observable=observable)
        else: # Implementation of finite shots
            observables_list = [observable]
            output = expval_joint_sampling_grouping(qdev, observables=observables_list, n_shots_per_group=self.n_shots)[observable]
        return output

    def forward(self, x, qdev):
        """
        Forward pass: Encode the input quantum state and classify using classical layers.
        """
        output = self.quantum_forward(x, qdev)  # Get the quantum output
        if self.n_labels > 2:
            output = self.fc(output.unsqueeze(-1)) # Pass through a fully connected layer for multi-class classification
        return output
    
def compute_accuracy(model, test_loader, scale, loss_lb, acc_thres):
    """
    Compute the accuracy of the model on the test dataset.
    Args:
        model (nn.Module): The trained model.
        test_loader (DataLoader): DataLoader for the test data.
        qdev (QuantumDevice): Quantum device used in the model.
    Returns:
        float: Accuracy of the model on the test dataset.
    """
    correct = 0
    total = 0
    loss = 0.0

    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient computation
        for data in test_loader:
            test_data_qpm, test_labels_qpm = data

            qdev = tq.QuantumDevice(n_wires=model.encoder.n_wires, bsz=test_data_qpm.size(0), device='cpu')
            # Forward pass through the model
            y_out = model(test_data_qpm, qdev)
            #print(f"Test Min Output: {y_out.min().item():.4f}, Max Output: {y_out.max().item():.4f}")

            if model.n_labels == 2:
                test_labels_qpm = test_labels_qpm.float()

            # Calculate loss
            loss += batch_loss(test_labels_qpm, y_out, n_labels=model.n_labels, loss_lb=loss_lb, scale=scale).item()

            if model.n_labels > 2:
                # In the case of multi-class classification
                _, predicted = torch.max(y_out, 1)
            else:
                # In the case of binary classification
                if 'logis' in loss_lb:
                    sg_y_out = torch.sigmoid(y_out * scale)
                else:
                    sg_y_out = y_out
                predicted = (sg_y_out > acc_thres).long()
            
            # Compare predictions with ground truth
            correct += (predicted == test_labels_qpm).sum().item()
            total += test_labels_qpm.size(0)

    accuracy = 100.0 * correct / total
    loss /= len(test_loader)
    return accuracy, loss

def batch_loss(target, predict, n_labels, loss_lb='mse', scale=1.0):
    if n_labels > 2:
        # Convert target labels to integers
        target = target.long()
        # Use cross-entropy loss for multi-class classification
        return F.cross_entropy(predict, target)
    else:
        # Use the original loss function for binary classification.
        ## TRAN (12/27): LET US USE MSE FOR NOW
        if loss_lb == 'logis':
            # Assume that the predict in [-1,1], target = 0 or 1
            # If we do not want predict moves to a position with too large magnitude
            # we can multiply with scale for soft margin. Larger scale can help to see the effect of Q-CurL
            sg_predict = torch.sigmoid(scale*predict)
            loss = - target * torch.log(sg_predict) - (1.0 - target) * torch.log(1.0 - sg_predict)
            return torch.mean(loss)
        elif loss_lb == 'fix_logis':
            # Assume that the predict in [-1,1], target = 0 or 1
            # Here, we do some tricky that maps target: 0 to 0.5, 1 to 1 to emphasize the output to 1
            # Since the predict tends to be zero at the beginning it means that sg_predict = 0.5 (with sigmoid)
            # Then we must modify the target as 0.5 for negative samples, just moving the positive samples to 1
            sg_target = (torch.sign(target) + 1.0) / 2.0
            
            # If we do not want predict moves to a position with too large magnitude
            # we can multiply with scale for soft margin. Larger scale can help to see the effect of Q-CurL
            sg_predict = torch.sigmoid(scale*predict)
            loss = - sg_target * torch.log(sg_predict) - (1.0 - sg_target) * torch.log(1.0 - sg_predict)
            return torch.mean(loss)
        elif loss_lb == 'smlogis':
            # Assume that the predict in [-1,1], target = 0 or 1
            sg_target = 0.9 * target + 0.05 
            sg_predict = torch.sigmoid(scale*predict)
            loss = - sg_target * torch.log(sg_predict) - (1.0 - sg_target) * torch.log(1.0 - sg_predict)
            return torch.mean(loss)
        else:
            err = predict - target
            return torch.mean(err**2)
