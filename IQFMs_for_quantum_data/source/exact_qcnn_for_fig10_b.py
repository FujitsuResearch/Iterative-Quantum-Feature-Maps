from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

import torchquantum as tq
import torchquantum.functional as tqf
from torchquantum.measurement import expval_joint_analytical, expval_joint_sampling
import random
import numpy as np
import matplotlib.pyplot as plt

from utils import *

import pickle

def _set_qdevice_state(qdev: tq.QuantumDevice, states: torch.Tensor) -> None:

    if hasattr(qdev, "set_states"):
        qdev.set_states(states)
        return

    # Fallbacks (older versions)
    if hasattr(qdev, "states"):
        qdev.states = states
        return

    raise AttributeError(
        "Could not set quantum state on QuantumDevice. "
    )

def apply_toffoli_decomposed(qdev, c1, c2, t):

    tqf.h(qdev, wires=t)
    tqf.cnot(qdev, wires=[c2, t])
    tqf.tdg(qdev, wires=t)
    tqf.cnot(qdev, wires=[c1, t])
    tqf.t(qdev, wires=t)
    tqf.cnot(qdev, wires=[c2, t])
    tqf.tdg(qdev, wires=t)
    tqf.cnot(qdev, wires=[c1, t])
    tqf.t(qdev, wires=c2)
    tqf.t(qdev, wires=t)
    tqf.cnot(qdev, wires=[c1, c2])
    tqf.h(qdev, wires=t)
    tqf.t(qdev, wires=c1)
    tqf.tdg(qdev, wires=c2)
    tqf.cnot(qdev, wires=[c1, c2])

def apply_toffoli_x_controls(
    qdev: tq.QuantumDevice,
    control1: int,
    control2: int,
    target: int,
) -> None:
    tqf.hadamard(qdev, wires=control1)
    tqf.hadamard(qdev, wires=control2)
    apply_toffoli_decomposed(qdev, control1, control2, target)
    tqf.hadamard(qdev, wires=control1)
    tqf.hadamard(qdev, wires=control2)

@dataclass(frozen=True)
class ExactQCNNConfig:
    n_wires: int = 9

class ExactQCNN9Open(nn.Module):

    def __init__(self, cfg: ExactQCNNConfig = ExactQCNNConfig()):
        super().__init__()
        if cfg.n_wires != 9:
            raise ValueError("This reference implementation is specialized to n_wires=9.")
        self.cfg = cfg

        # Cache kept qubits for N=9, d=1
        self.blocks = [(0,1,2), (3,4,5), (6,7,8)]
        self.kept = [1,4,7] 

    def _apply_convolution(self, qdev: tq.QuantumDevice) -> None:
        n = self.cfg.n_wires

        # Nearest-neighbor CZ in two layers (brickwork), matching the diagram.
        for i in range(0, n - 1, 2):
            tqf.cz(qdev, wires=[i, i + 1])
        for i in range(1, n - 1, 2):
            tqf.cz(qdev, wires=[i, i + 1])

        # Long-range CZ between neighboring kept qubits (distance 3 in the original chain)
        for a, b in zip(self.kept[:-1], self.kept[1:]):
            tqf.cz(qdev, wires=[a, b])

        # block = (left, mid, right)
        for left, mid, right in self.blocks:
            apply_toffoli_x_controls(qdev, left, right, mid)

        for i in range(2, n - 1, 3):
            tqf.swap(qdev, wires=[i, i + 1])


    # pooling
    def _apply_pooling(self, qdev: tq.QuantumDevice) -> None:
        n = self.cfg.n_wires

        for i in range(0, n - 1, 3):
            tqf.h(qdev, wires=i)
            tqf.cz(qdev, wires=[i, i + 1])

        for i in range(2, n, 3):
            tqf.h(qdev, wires=i)
            tqf.cz(qdev, wires=[i, i - 1])

    #FC
    def _apply_fc(self, qdev: tq.QuantumDevice) -> None:
        for a, b in zip(self.kept[:-1], self.kept[1:]):
            tqf.cz(qdev, wires=[a, b])

    def _observable_str_for_wire(self, wire: int) -> str:
        obs = ["I"] * self.cfg.n_wires
        obs[wire] = "X"
        return "".join(obs)

    def forward(self, states, n_shots):

        bsz = states.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.cfg.n_wires, bsz=bsz, device=states.device)
        _set_qdevice_state(qdev, states)

        # Apply circuit: (C -> P) x d  with d=1 here, then FC
        self._apply_convolution(qdev)
        self._apply_pooling(qdev)
        self._apply_fc(qdev)

        readout_wires = [self.kept[1]]  # middle of {kept0, kept1, kept2}

        obs_list = [self._observable_str_for_wire(w) for w in readout_wires]

        def _measure_once() -> torch.Tensor:
            # returns shape (bsz,) if one obs, else (bsz, len(obs_list))
            vals = []

            for obs in obs_list:
                if n_shots is None:
                    v = expval_joint_analytical(qdev, obs)
                else:
                    v = expval_joint_sampling(qdev, obs, n_shots=n_shots)
                vals.append(v)

            out = torch.stack(vals, dim=-1)
            if out.shape[-1] == 1:
                out = out.squeeze(-1)
            return out

        return _measure_once()


class GroundStatePKLDataset(Dataset):
    def __init__(self, pkl_path, device="cpu"):
        with open(pkl_path, "rb") as f:
            self.data = pickle.load(f)  # list of dicts: h1,h2,energy,ground_state
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # ground_state: numpy complex vector (2**n,)
        gs = item["ground_state"]
        # torchquantum expects complex state tensor; batch dimension will be added by DataLoader
        x = torch.tensor(gs, dtype=torch.complex64)
        h2 = float(item["h2"])
        return x, h2 


if __name__ == "__main__":
    n = 9
    device = "cpu"

    data_dir = "../data/phase_detection_python_open_9qubit_h1_0.5"
    data_prefix = "ground_nq_9_h1_0.5000_h2_50pts"
    pkl_path = os.path.join(data_dir, f"{data_prefix}_data.pkl")

    test_ds = GroundStatePKLDataset(pkl_path, device=device)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    model = ExactQCNN9Open(ExactQCNNConfig())

    seed_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    shot_list = [10, 1000]

    results_per_shot = {}

    for n_shots in shot_list:
        print(f"\n===== n_shots = {n_shots} =====")

        all_outputs = [] 
        h2_list_ref = None

        for seed in seed_list:
            np.random.seed(seed)
            torch.manual_seed(seed)
            random.seed(seed)

            outputs = []
            h2_list = []
            for x, h2 in test_loader:
                y_trial = model(x, n_shots=n_shots)  
                outputs.append(float(y_trial.item()))
                h2_list.append(float(h2.item()))

            print(f"seed={seed}  outputs(head)={outputs[:5]}")

            all_outputs.append(outputs)

            if h2_list_ref is None:
                h2_list_ref = h2_list

        # list → numpy array: (num_seeds, num_data)
        all_outputs_np = np.array(all_outputs)  # shape (num_seeds, num_data)

        mean_outputs_np = np.mean(all_outputs_np, axis=0)  # shape (num_data,)
        std_outputs_np = np.std(all_outputs_np, axis=0, ddof=1) # shape (num_data,)

        results_per_shot[n_shots] = {
            "h2_list": np.array(h2_list_ref),
            "mean_outputs": mean_outputs_np,
            "std_outputs": std_outputs_np,
            "all_outputs": all_outputs_np,
        }

        print(f"[n_shots={n_shots}] mean_outputs(head)={mean_outputs_np[:5].tolist()}")
        print(f"[n_shots={n_shots}] std_outputs(head)={std_outputs_np[:5].tolist()}")

    save_path = "../post_process/exact_qcnn_results_per_shot.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(results_per_shot, f)

    print(f"Saved results_per_shot to {save_path}")
