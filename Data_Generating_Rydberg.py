import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
import jax
import logging
import sys
import netket as nk
from netket.operator.spin import sigmax, sigmaz, sigmam, sigmap
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import random
import time
from collections import defaultdict
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix
from concurrent.futures import ProcessPoolExecutor, as_completed

# -----------------------------------------------------
# Set up logging
# -----------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rydberg_system.log", mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------
# Constants
# -----------------------------------------------------
Nx = 2             
R_cut = 6.0 

def reshape_for_subsystem(psi, A_indices, N):
    """Reshape wavefunction for bipartition"""
    A_indices = sorted(A_indices)
    B_indices = [i for i in range(N) if i not in A_indices]
    N_A = len(A_indices)
    N_B = N - N_A

    psi_reshaped = np.zeros((2**N_A, 2**N_B), dtype=psi.dtype)

    A_pos_map = {spin: pos for pos, spin in enumerate(A_indices)}
    B_pos_map = {spin: pos for pos, spin in enumerate(B_indices)}

    for i in range(2**N):
        i_bin = i
        i_A = 0
        i_B = 0
        for spin in range(N):
            bit = i_bin & 1
            i_bin >>= 1
            if spin in A_pos_map:
                i_A |= (bit << A_pos_map[spin])
            else:
                i_B |= (bit << B_pos_map[spin])

        psi_reshaped[i_A, i_B] = psi[i]

    return psi_reshaped


def calculate_classical_MI(psi_gs, A_indices, N):
    # Get subsystem B indices
    B_indices = [i for i in range(N) if i not in A_indices]
    
    # Calculate joint probabilities
    p_joint = np.abs(psi_gs) ** 2
    
    # Calculate marginal probabilities
    p_A = np.zeros(2**len(A_indices))
    p_B = np.zeros(2**len(B_indices))
    
    for state_idx in range(2**N):
        state_bin = format(state_idx, f'0{N}b')
        A_state = int(''.join(state_bin[i] for i in A_indices), 2)
        B_state = int(''.join(state_bin[i] for i in B_indices), 2)
        p_A[A_state] += p_joint[state_idx]
        p_B[B_state] += p_joint[state_idx]
    
    # Calculate classical mutual information
    MI = 0
    for state_idx in range(2**N):
        if p_joint[state_idx] > 1e-12:  # Numerical stability
            state_bin = format(state_idx, f'0{N}b')
            A_state = int(''.join(state_bin[i] for i in A_indices), 2)
            B_state = int(''.join(state_bin[i] for i in B_indices), 2)
            MI += p_joint[state_idx] * np.log(p_joint[state_idx] / (p_A[A_state] * p_B[B_state] + 1e-12))
    
    return float(MI)

def calculate_properties_cpu_batch(params_batch):
    batch_results = []

    if not params_batch:
        logger.warning("Received an empty batch. Returning.")
        return batch_results

    Ny_values = [params[0] for params in params_batch]
    if len(set(Ny_values)) != 1:
        logger.warning("Inconsistent Ny values in the batch.")
        return batch_results

    Ny = Ny_values[0]
    N = Nx * Ny

    # Define Hilbert Space
    hi = nk.hilbert.Spin(s=1/2, N=N)

    # Precompute positions
    positions_list = []
    for params in params_batch:
        positions = np.array([
            (col * 1, row * 2)
            for row in range(Nx) for col in range(Ny)
        ])
        positions_list.append(positions)

    H_list = []

    # Construct Hamiltonians
    for idx, params in enumerate(params_batch):
        Ny_current, Delta_over_Omega, Rb_over_a = params
        positions = positions_list[idx]

        dx = positions[:, 0][:, np.newaxis] - positions[:, 0]
        dy = positions[:, 1][:, np.newaxis] - positions[:, 1]
        r_squared = dx.astype(float)**2 + dy.astype(float)**2
        r_squared[r_squared == 0] = np.inf
        r = np.sqrt(r_squared)
        interaction_mask = (r <= R_cut)
        interaction_mask = np.triu(interaction_mask, k=1)
        valid_pairs = np.argwhere(interaction_mask)

        V_ij = Rb_over_a**6 / (4 * r_squared[interaction_mask] ** 3)

        H = nk.operator.LocalOperator(hi)

        sigmap_ops = [sigmap(hi, i) for i in range(N)]
        sigmam_ops = [sigmam(hi, i) for i in range(N)]
        sigmaz_ops = [sigmaz(hi, i) for i in range(N)]

        for i in range(N):
            H += (1 / 2) * (sigmap_ops[i] + sigmam_ops[i])
            H += (Delta_over_Omega / 2) * (sigmaz_ops[i] - 1)

        for (i, j), V in zip(valid_pairs, V_ij):
            H += V * (sigmaz_ops[i] - 1) * (sigmaz_ops[j] - 1)

        sp_h = H.to_sparse()
        sp_h_cpu = csr_matrix(sp_h)
        H_list.append(sp_h_cpu)

    # Compute eigenvalues
    eig_vals = []
    eig_vecs = []
    for H in H_list:
        try:
            val, vec = eigsh(H, k=1, which="SA")
            eig_vals.append(val)
            eig_vecs.append(vec)
        except Exception as e:
            logger.exception(f"Error during eigenvalue computation: {e}")
            eig_vals.append(None)
            eig_vecs.append(None)

    # Process results
    for idx, (params, E_gs, psi_gs) in enumerate(zip(params_batch, eig_vals, eig_vecs)):
        if E_gs is None or psi_gs is None:
            continue

        Ny_current, Delta_over_Omega, Rb_over_a = params

        # Normalize wavefunction
        psi_gs = psi_gs[:, 0]
        psi_gs /= np.linalg.norm(psi_gs)

        probabilities = np.abs(psi_gs) ** 2
        indices = list(range(len(probabilities)))

        A_size = random.randint(1, N - 1)
        A_indices = random.sample(range(N), A_size)
        A_mask = np.zeros(N, dtype=int)
        A_mask[A_indices] = 1
        A_mask_str = ''.join(str(x) for x in A_mask)

        psi_gs_reshaped = reshape_for_subsystem(psi_gs, A_indices, N)
        
        U, s, Vh = np.linalg.svd(psi_gs_reshaped, full_matrices=False)
        s_squared = s**2
        s_squared_normalized = s_squared / np.sum(s_squared)
        entropy = -np.sum(s_squared_normalized * np.log(s_squared_normalized + 1e-12))
        
        classical_MI = calculate_classical_MI(psi_gs, A_indices, N)
        batch_results.append({
            'Ny': Ny_current,
            'Delta_over_Omega': Delta_over_Omega,
            'Rb_over_a': Rb_over_a,
            'Energy': float(E_gs[0]),
            'All_Indices': indices,
            'All_Probabilities': probabilities.tolist(),
            'Von_Neumann_Entropy': float(entropy),
            'Classical_MI': float(classical_MI),
            'N_A': A_size,
            'Subsystem_Mask': A_mask_str
        })

    return batch_results

# -----------------------------------------------------
# Parameters
# -----------------------------------------------------
Ny_range = (1, 6)             
Delta__over_Omega_range = (0.0, 4.0)         
Rb_over_a_range = (0.1, 5.0)  
num_points = 10000           

np.random.seed(42)

parameter_list = [
    (
        np.random.randint(Ny_range[0], Ny_range[1] + 1),
        np.random.uniform(Delta__over_Omega_range[0], Delta__over_Omega_range[1]),
        np.random.uniform(Rb_over_a_range[0], Rb_over_a_range[1])
    )
    for _ in range(num_points)
]

grouped_params = defaultdict(list)
for params in parameter_list:
    Ny = params[0]
    grouped_params[Ny].append(params)

if __name__ == "__main__":
    logger.info("Starting the main execution.")
    
    schema = pa.schema([
    ('Ny', pa.int64()),
    ('Delta_over_Omega', pa.float64()),
    ('Rb_over_a', pa.float64()),
    ('Energy', pa.float64()),
    ('All_Indices', pa.list_(pa.int64())),
    ('All_Probabilities', pa.list_(pa.float64())),
    ('Von_Neumann_Entropy', pa.float64()),
    ('Classical_MI', pa.float64()),
    ('N_A', pa.int64()),
    ('Subsystem_Mask', pa.string())
    ])

    all_param_batches = []
    for Ny, params_list in grouped_params.items():
        N = Nx * Ny
        if N <= 4:
            batch_size = 10000
        elif N <= 12:
            batch_size = 500
        else:
            batch_size = 5

        for i in range(0, len(params_list), batch_size):
            batch_slice = params_list[i:i + batch_size]
            all_param_batches.append(batch_slice)

    writer = pq.ParquetWriter('Rydberg_data.parquet', schema)
    total_batches = len(all_param_batches)
    start_time_all = time.time()

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(calculate_properties_cpu_batch, batch) for batch in all_param_batches]
        completed = 0

        for future in as_completed(futures):
            try:
                batch_results = future.result()
            except Exception as e:
                logger.exception(f"Exception raised by a worker process: {e}")
                continue

            if batch_results:
                df_batch = pd.DataFrame(batch_results)
                if not df_batch.empty:
                    table = pa.Table.from_pandas(df_batch, schema=schema)
                    writer.write_table(table)

            completed += 1
            logger.info(f"Completed batch {completed} of {total_batches}")

    writer.close()
    end_time_all = time.time()
    logger.info(f"All batches processed in {end_time_all - start_time_all:.2f} seconds.")