# Iterative-Quantum-Feature-Maps (IQFMs)
This repository contains the source codes for the Iterative-Quantum-Feature-Maps (IQFMs), a hybrid quantum-classical framework that constructs a deep architecture by iteratively connecting shallow QFMs with classically computed augmentation weights.
It has been introduced in the paper titled "Iterative Quantum Feature Maps".

Link: https://arxiv.org/abs/2506.19461

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


## Installation (Notes for Miniconda / conda users)

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