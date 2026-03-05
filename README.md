# Iterative-Quantum-Feature-Maps (IQFMs)
This repository contains the source codes for the Iterative-Quantum-Feature-Maps (IQFMs), a hybrid quantum-classical framework that constructs a deep architecture by iteratively connecting shallow QFMs with classically computed augmentation weights.
It has been introduced in the paper titled "Iterative Quantum Feature Maps".

Link: https://arxiv.org/abs/2506.19461

# Repository Structure

The repository is organized as follows:

```text
Iterative-Quantum-Feature-Maps/
├── IQFMs_for_classical_data/                           # Folder for classical data 
│   ├── postprocess                                     # Functions for plotting and analyzing experimental results
│   ├── runscripts          　                          # Shell scripts to execute the experiments
│   └── source                                          # Source code for IQFMs and classical NN
│
├── IQFMs_for_quantum_data/                             # Folder for quantum data 
│   ├── data
│   │　 ├── phase_detection_gch_nq_8                    # TaskB data
│   │　 ├── phase_detection_python_open_9qubit          # TaskA data (for Fig.10(a))
│   │　 ├── phase_detection_python_open_9qubit_h1_0.5   # TaskA data (for Fig.10(b))
│   │　 └── phase_detection_python_periodic             # TaskA data (for other figures)
│   ├── post_process                                    # Functions for plotting and analyzing experimental results 
│   ├── runscripts                                      # Shell scripts to execute the experiments
│   └── source                                          # Source code for IQFMs, QCNN, exactQCNN
│
└── shadow_kernel/                                      # Folder for shadow kernel method 
    ├── results                                         # Results of quantum phase classification
    ├── Data_dmrg.jl                                    # Ground state calculation with DMRG & obtain classical shadows
    ├── Model_SK.jl                                     # Calculate the Gram matrix of the shadow kernel from the classical shadow 
    ├── SVM_SK.ipynb                                    # Quantum phase classification with SVM from Gram matrix
    └── my_cs_tools_v250625.jl                          # A file that summarizes functions related to classic shadows
```

# Installation

1. **Clone the repository**
```bash
git clone https://github.com/FujitsuResearch/Iterative-Quantum-Feature-Maps.git
cd Iterative-Quantum-Feature-Maps
```

2. **Install the required packages for Python (tested with Python v3.10.12 on Linux)**
```bash
# Create an environment
python3 -m venv .venv

# Activate the environment
source .venv/bin/activate

# Install packages
pip install -r requirements.txt
```

**Installation (Notes for Miniconda users)**

If you prefer to use Miniconda, you can create and activate the environment as follows:

```bash
conda create -n .venv python=3.10.12
conda activate .venv
pip install -r requirements.txt
```

In some environments, running pip install -r requirements.txt may result in the following error:

ModuleNotFoundError: No module named 'pkg_resources'

If this happens, downgrading setuptools may resolve the issue:
```bash
pip install "setuptools<80" -v
```
After that, please run pip install -r requirements.txt again.


# Running the Experiments

When performing IQFMs experiments, please run them as shown in the following examples.

```bash
cd IQFMs_for_quantum_data/runscripts
sh fig5_a_iqfms_contrastive_learning_taskA.sh
```

```bash
cd IQFMs_for_classical_data/runscripts
sh fig9_iqfms_contrastive_learning.sh
```

# Post-processing Results

When analyzing experimental results of IQFMs, please  run the following commands as shown in the following examples.
You can output information on the average accuracy and standard deviation for each condition.

```bash
cd IQFMs_for_quantum_data/postprocess
python acc_stats.py --folder '../results/fig5_a_iqfms_contrastive_learning_taskA'
```

```bash
cd IQFMs_for_classical_data/postprocess
python acc_stats.py --folder '../results/fig9_iqfms_contrastive_learning'
```