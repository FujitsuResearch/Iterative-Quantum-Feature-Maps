import os
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sp_linalg
import argparse
import pickle

# Define Pauli matrices
II = sparse.identity(2, format="csr")
Z = sparse.diags([1, -1], 0, shape=(2, 2), dtype=np.complex64)
X = sparse.csr_matrix([[0, 1], [1, 0]], dtype=np.complex64)


def kronecker_product(op, index, num_qubits):
    """
    Efficiently computes the Kronecker product for a given operator inserted at the specified index.
    """
    ops = [II] * num_qubits
    ops[index] = op
    result = ops[0]
    for i in range(1, num_qubits):
        result = sparse.kron(result, ops[i], format="csr")
    return result


class Hamiltonian:
    def __init__(self, num_qubits=4, h1_range=(0, 1.6), h2_range=(-1.6, 1.6), verbose=True):
        self.num_qubits = num_qubits
        self.size = 2 ** num_qubits
        self.verbose = verbose
        self.h1_min, self.h1_max = h1_range
        self.h2_min, self.h2_max = h2_range

        # Initialize Hamiltonian terms as sparse matrices
        self.first_term = sparse.csr_matrix((self.size, self.size), dtype=np.complex64)
        self.second_term = sparse.csr_matrix((self.size, self.size), dtype=np.complex64)
        self.third_term = sparse.csr_matrix((self.size, self.size), dtype=np.complex64)
       

    def construct_first_term(self):
        """Constructs the first term of the Hamiltonian: -J * sum_i Z_i Z_{i+2}."""
        for i in range(self.num_qubits):
            if self.verbose:
                print(f"Constructing first term {i+1}/{self.num_qubits}")
            a = kronecker_product(Z, i, self.num_qubits)
            b = kronecker_product(X, (i + 1) % self.num_qubits, self.num_qubits)
            c = kronecker_product(Z, (i + 2) % self.num_qubits, self.num_qubits)
            combined = a.dot(b).dot(c)
            self.first_term -= combined

    def construct_second_term(self):
        """Constructs the second term of the Hamiltonian: -h1 * sum_i X_i."""
        for i in range(self.num_qubits):
            if self.verbose:
                print(f"Constructing second term {i+1}/{self.num_qubits}")
            self.second_term -= kronecker_product(X, i, self.num_qubits)

    def construct_third_term(self):
        """Constructs the third term of the Hamiltonian: -h2 * sum_i X_i X_{i+1}."""
        for i in range(self.num_qubits):
            if self.verbose:
                print(f"Constructing third term {i+1}/{self.num_qubits - 1}")
            b1 = kronecker_product(X, i, self.num_qubits)
            b2 = kronecker_product(X, (i + 1) % self.num_qubits, self.num_qubits)
            self.third_term -= b1.dot(b2)

    def construct_hamiltonian(self, h1, h2):
        """Constructs the full Hamiltonian for given h1 and h2."""
        return self.first_term + h1 * self.second_term + h2 * self.third_term

    def compute_ground_state(self, h1, h2):
        """Finds the ground state and corresponding energy of the Hamiltonian."""
        hamiltonian = self.construct_hamiltonian(h1, h2)
        eigenvalues, eigenvectors = sp_linalg.eigsh(hamiltonian, k=1, which='SA')
        return eigenvalues[0].real, eigenvectors[:, 0]

    def generate_data(self, save_dir, h1_steps, h2_steps, filename_prefix, h1_border=None, h2_border=None):
        """
        Generates the ground state data over specified h1 and h2 ranges.
        Saves the data to a pickle file to reduce memory usage.
        """
        self.construct_first_term()
        self.construct_second_term()
        self.construct_third_term()

        h1_vals = np.linspace(self.h1_min, self.h1_max, h1_steps)
        h2_vals = np.linspace(self.h2_min, self.h2_max, h2_steps)

        data = []
        labels = []
        for h1 in h1_vals:
            for h2 in h2_vals:
                if h2_border is not None:
                    h2 = h2_border
                if h1_border is not None:
                    # Label = 1 in SPT phase, other is 0
                    if h2 > -1.0:
                        label = 1 if h1 <= h1_border else 0
                    else:
                        label = 1 if h1 > h1_border else 0
                        
                    labels.append(label)

                energy, ground_state = self.compute_ground_state(h1, h2)
                data.append({"h1": h1, "h2": h2, "energy": energy, "ground_state": ground_state})

        filename = f"{filename_prefix}_data.pkl"
        labels_filename = f"{filename_prefix}_labels.pkl"

        with open(os.path.join(save_dir, filename), 'wb') as f:
            pickle.dump(data, f)

        if labels:
            with open(os.path.join(save_dir, labels_filename), 'wb') as f:
                pickle.dump(labels, f)
        print(f"Data saved to {filename}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='../data/phase_detection_python_periodic')
    parser.add_argument('--num_qubits', type=int, default=8)
    parser.add_argument('--train_steps', type=int, default=40)
    parser.add_argument('--steps', type=int, default=64)
    parser.add_argument('--gen_all', type=int, default=1)
    args = parser.parse_args()

    save_dir, num_qubits = args.save_dir, args.num_qubits
    n_train, steps = args.train_steps, args.steps
    n_test = n_train

    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    hamiltonian = Hamiltonian(num_qubits=args.num_qubits)
    hamiltonian.generate_data(save_dir, n_train, 1, f"ground_nq_{num_qubits}_train_{n_train}", h1_border=1.0, h2_border=0.0)

    # Define specific phase boundaries for test data generation
    h1_vals = [0.1000, 0.2556, 0.4111, 0.5667, 0.7222, 0.8778, 1.0333, 1.1889, 1.3444, 1.5000]
    h2_vals_para = [0.8439, 0.6636, 0.5033, 0.3631, 0.2229, 0.09766, -0.02755, -0.1377, -0.2479, -0.3531]
    h2_vals_anti_ferro = [-1.004, -1.0009, -1.024, -1.049, -1.079, -1.109, -1.154, -1.225, -1.285, -1.35]

    # Generate test data based on the phase boundaries
    for idx, (h1_border, h2_border) in enumerate(zip(h1_vals, h2_vals_para)):
        filename_prefix = f"ground_nq_{num_qubits}_test_{n_test}_para_{idx}"
        hamiltonian.generate_data(save_dir, n_test, 1, filename_prefix, h1_border=h1_border, h2_border=h2_border)

    for idx, (h1_border, h2_border) in enumerate(zip(h1_vals, h2_vals_anti_ferro)):
        filename_prefix = f"ground_nq_{num_qubits}_test_{n_test}_anti_ferro_{idx}"
        hamiltonian.generate_data(save_dir, n_test, 1, filename_prefix, h1_border=h1_border, h2_border=h2_border)


    # Generate diagram data
    if args.gen_all > 0:
        hamiltonian.generate_data(args.save_dir, steps, steps, f"ground_nq_{num_qubits}_diagram_{steps}_{steps}")
