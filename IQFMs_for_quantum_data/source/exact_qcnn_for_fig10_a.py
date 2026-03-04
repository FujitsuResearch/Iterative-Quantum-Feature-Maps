from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

import torchquantum as tq
import torchquantum.functional as tqf
from torchquantum.measurement import expval_joint_analytical, expval_joint_sampling
import random
import numpy as np

from utils import *

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
        """Apply the convolution part (blue C layer in Fig.2(b))."""
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
        """Apply the pooling part (orange P layer in Fig.2(b)).
        """
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

if __name__ == "__main__":
    n = 9
    train_dat_type = 'all'
    test_dat_type = 'all'
    _, test_loader = read_IsingCluster_data_open(n, train_dat_type, test_dat_type=test_dat_type)

    device ="cpu"

    model = ExactQCNN9Open(ExactQCNNConfig())

    seed_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    shot_list = [10, 100, 500, 1000]

    for n_shots in shot_list:
        print(f"\n===== n_shots = {n_shots} =====")
        acc_trials = []

        for seed in seed_list:
            np.random.seed(seed)
            torch.manual_seed(seed)
            random.seed(seed)

            correct = 0
            total = 0

            for x, y in test_loader:
                y_trial = model(x, n_shots=n_shots)
                pred = (y_trial > 0.5).long()

                correct += (pred == y).sum().item()
                total += y.numel()

            acc = correct / total
            acc_trials.append(acc)

        acc_mean = float(np.mean(acc_trials))
        acc_std = float(np.std(acc_trials, ddof=1)) if len(acc_trials) > 1 else 0.0
        print('acc_trials', acc_trials)
        print('mean accuracy:',acc_mean)
        print('std accuracy:', acc_std)