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
    def __init__(self, num_qubits=9, h1_range=(0, 1.6), h2_range=(-1.6, 1.6), verbose=True):
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
        """Constructs the first term of the Hamiltonian: -J * sum_i Z_i X_{i+1} Z_{i+2}."""
        for i in range(self.num_qubits - 2):
            if self.verbose:
                print(f"Constructing first term {i+1}/{self.num_qubits - 2}")
            a = kronecker_product(Z, i, self.num_qubits)
            b = kronecker_product(X, i + 1, self.num_qubits)
            c = kronecker_product(Z, i + 2, self.num_qubits)
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
        for i in range(self.num_qubits - 1):  # OBC
            if self.verbose:
                print(f"Constructing third term {i+1}/{self.num_qubits - 1}")
            b1 = kronecker_product(X, i, self.num_qubits)
            b2 = kronecker_product(X, i + 1, self.num_qubits)
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
    parser = argparse.ArgumentParser(description='Generate ground-state data with fixed h1 and scanned h2.')
    parser.add_argument('--save_dir', type=str, default='../data/phase_detection_python_open_9qubit_h1_0.5')
    parser.add_argument('--num_qubits', type=int, default=9)

    # Requested setting: fix h1 and take evenly spaced h2 points
    parser.add_argument('--h1', type=float, default=0.5, help='Fixed value of h1')
    parser.add_argument('--h2_min', type=float, default=-1.6, help='Minimum value of h2')
    parser.add_argument('--h2_max', type=float, default=1.6, help='Maximum value of h2')
    parser.add_argument('--h2_points', type=int, default=50, help='Number of equally spaced h2 points')

    # Optional: turn off verbose prints
    parser.add_argument('--verbose', type=int, default=1, help='1: verbose, 0: silent')

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Fix h1 by setting its range to (h1, h1) and use 1 step.
    hamiltonian = Hamiltonian(
        num_qubits=args.num_qubits,
        h1_range=(args.h1, args.h1),
        h2_range=(args.h2_min, args.h2_max),
        verbose=bool(args.verbose),
    )

    filename_prefix = f"ground_nq_{args.num_qubits}_h1_{args.h1:.4f}_h2_{args.h2_points}pts"
    hamiltonian.generate_data(
        save_dir=args.save_dir,
        h1_steps=1,
        h2_steps=args.h2_points,
        filename_prefix=filename_prefix,
        h1_border=None,
        h2_border=None,
    )
