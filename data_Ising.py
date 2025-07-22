import cupy as cp
import cupyx.scipy.sparse as csp
import cupyx.scipy.sparse.linalg as cx
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
import pandas as pd
import random
import time
import logging
import sys
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

# -----------------------------------------------------
# Set up logging
# -----------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("transverse_ising_model.log", mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def precompute_operators(N_s):
    """
    Precompute all necessary operators for a system of size N_s.
    Returns sigma_z_ops and sigma_x_pair_ops that can be reused for all Hamiltonians of this size.
    """
    I = csp.eye(2, dtype=cp.float32)
    sigma_z = csp.csr_matrix(cp.array([[1, 0], [0, -1]], dtype=cp.float32))
    sigma_x = csp.csr_matrix(cp.array([[0, 1], [1, 0]], dtype=cp.float32))
    
    sigma_z_ops = []
    for i in range(N_s):
        op = csp.identity(1, dtype=cp.float32)
        for j in range(N_s):
            op = csp.kron(op, sigma_z if j == i else I)
        sigma_z_ops.append(op)
    
    sigma_x_pair_ops = []
    for i in range(N_s - 1):
        op = csp.identity(1, dtype=cp.float32)
        for j in range(N_s):
            op = csp.kron(op, sigma_x if (j == i or j == i + 1) else I)
        sigma_x_pair_ops.append(op)
    
    return sigma_z_ops, sigma_x_pair_ops

def make_hamiltonian(N_s, J_over_h, sigma_z_ops, sigma_x_pair_ops):
    """
    Create the transverse Ising model Hamiltonian:
      H = -Σ σᶻᵢ - J_over_h * Σ σˣᵢσˣᵢ₊₁
    """
    H = csp.csr_matrix((2**N_s, 2**N_s), dtype=cp.float32)
    for op in sigma_z_ops:
        H -= op
    for op in sigma_x_pair_ops:
        H -= J_over_h * op
    return H

def reshape_for_subsystem(psi, A_indices, N):
    A_indices = sorted(A_indices)
    B_indices = sorted([i for i in range(N) if i not in A_indices])
    N_A, N_B = len(A_indices), N - len(A_indices)
    
    states = cp.arange(2**N, dtype=cp.uint32)
    bits = ((states[:, None] >> cp.arange(N, dtype=cp.uint32)) & 1).astype(cp.int32)
    
    weights_A = cp.array([1 << i for i in range(N_A)], dtype=cp.int32)
    weights_B = cp.array([1 << i for i in range(N_B)], dtype=cp.int32)
    a_indices = cp.dot(bits[:, A_indices], weights_A)
    b_indices = cp.dot(bits[:, B_indices], weights_B)
    
    psi_reshaped = cp.zeros((2**N_A, 2**N_B), dtype=psi.dtype)
    psi_reshaped[a_indices, b_indices] = psi
    return psi_reshaped

def calculate_entropy(psi_gs, A_indices, N):
    psi_reshaped = reshape_for_subsystem(psi_gs, A_indices, N)
    _, s, _ = cp.linalg.svd(psi_reshaped, full_matrices=False)
    s2 = s**2
    s2_norm = s2 / cp.sum(s2)
    entropy = -cp.sum(s2_norm * cp.log(s2_norm + 1e-12))
    return float(entropy)

def calculate_classical_MI(psi_gs, A_indices, N):
    B_indices = [i for i in range(N) if i not in A_indices]
    p_joint = cp.abs(psi_gs)**2
    
    states = cp.arange(2**N, dtype=cp.uint32)
    bits = ((states[:, None] >> cp.arange(N, dtype=cp.uint32)) & 1).astype(cp.int32)
    
    N_A = len(A_indices)
    N_B = N - N_A
    weights_A = cp.array([1 << i for i in range(N_A)], dtype=cp.int32)
    weights_B = cp.array([1 << i for i in range(N_B)], dtype=cp.int32)
    a_indices = cp.dot(bits[:, A_indices], weights_A)
    b_indices = cp.dot(bits[:, B_indices], weights_B)
    
    p_A = cp.bincount(a_indices, weights=p_joint, minlength=2**N_A)
    p_B = cp.bincount(b_indices, weights=p_joint, minlength=2**N_B)
    
    ratio = p_joint / (p_A[a_indices] * p_B[b_indices] + 1e-12)
    MI = cp.sum(p_joint * cp.log(ratio + 1e-12))
    return float(MI)

def solve_and_analyze_batch_same_size(N_s, J_values, precomputed_operators):
    """
    Solve Hamiltonians and calculate properties for a batch of parameters of the same system size.
    """
    batch_results = []
    sigma_z_ops, sigma_x_pair_ops = precomputed_operators


    stream = cp.cuda.Stream()
    with stream:
        for J_over_h in J_values:
            try:
                H = make_hamiltonian(N_s, J_over_h, sigma_z_ops, sigma_x_pair_ops)
                E_gs, psi_gs = cx.eigsh(H, k=1, which="SA")
                psi_gs = psi_gs[:, 0]
                psi_gs /= cp.linalg.norm(psi_gs)
                probabilities = cp.abs(psi_gs)**2
                indices = list(range(int(probabilities.shape[0])))
    
                A_size = random.randint(1, N_s - 1)
                A_indices = random.sample(range(N_s), A_size)
                A_mask = ['1' if i in A_indices else '0' for i in range(N_s)]
                A_mask_str = ''.join(A_mask)
    
                entropy = calculate_entropy(psi_gs, A_indices, N_s)
                classical_MI = calculate_classical_MI(psi_gs, A_indices, N_s)
    
                batch_results.append({
                    'N_s': N_s,
                    'J_over_h': J_over_h,
                    'Energy': float(E_gs[0]),
                    'All_Indices': indices,
                    'All_Probabilities': cp.asnumpy(probabilities).tolist(),
                    'Von_Neumann_Entropy': entropy,
                    'Classical_MI': classical_MI,
                    'N_A': A_size,
                    'Subsystem_Mask': A_mask_str
                })
            except Exception as e:
                logger.exception(f"Error processing N_s={N_s}, J_over_h={J_over_h}: {e}")
    return batch_results

def process_group(args):
    N_s, J_values = args
    try:
        logger.info(f"Processing system size N_s={N_s} with {len(J_values)} J/h values")
        precomputed_operators = precompute_operators(N_s)
        results = solve_and_analyze_batch_same_size(N_s, J_values, precomputed_operators)
        return (N_s, results)
    except Exception as e:
        logger.exception(f"Error processing system size N_s={N_s}: {e}")
        return (N_s, [])

def main():
    # -----------------------------------------------------
    # Parameters
    # -----------------------------------------------------
    N_s_range = (2, 12)              # System size range
    J_over_h_range = (0.0, 2.0)        # J/h coupling ratio range
    num_points = 1000000             # Number of parameter combinations to sample

    random.seed(42)
    np.random.seed(42)
    
    parameter_list = [
        (random.randint(N_s_range[0], N_s_range[1]),
         random.uniform(J_over_h_range[0], J_over_h_range[1]))
        for _ in range(num_points)
    ]
    
    grouped_params = defaultdict(list)
    for N_s, J_over_h in parameter_list:
        grouped_params[N_s].append(J_over_h)
    
    schema = pa.schema([
        ('N_s', pa.int64()),
        ('J_over_h', pa.float64()),
        ('Energy', pa.float64()),
        ('All_Indices', pa.list_(pa.int64())),
        ('All_Probabilities', pa.list_(pa.float64())),
        ('Von_Neumann_Entropy', pa.float64()),
        ('Classical_MI', pa.float64()),
        ('N_A', pa.int64()),
        ('Subsystem_Mask', pa.string())
    ])
    
    parquet_filename = 'transverse_ising_data.parquet'
    
    logger.info(f"Starting calculations for {num_points} parameter combinations across {len(grouped_params)} different system sizes")
    start_time_all = time.time()
    
    all_results = []
    with ProcessPoolExecutor() as executor:
        future_to_group = {executor.submit(process_group, (N_s, J_values)): N_s for N_s, J_values in grouped_params.items()}
        for future in as_completed(future_to_group):
            N_s = future_to_group[future]
            try:
                group_N_s, group_results = future.result()
                if group_results:
                    df_group = pd.DataFrame(group_results)
                    temp_filename = f'temp_N_s_{group_N_s}.parquet'
                    df_group.to_parquet(temp_filename, index=False)
                    logger.info(f"Saved {len(group_results)} results for N_s={group_N_s} to {temp_filename}")
                    all_results.append((group_N_s, temp_filename, len(group_results)))
                else:
                    logger.warning(f"No results for system size N_s={group_N_s}")
            except Exception as e:
                logger.exception(f"Error in future for N_s={N_s}: {e}")
    
    logger.info("Combining all results into the final Parquet file...")
    try:
        all_dfs = []
        for N_s, filename, count in all_results:
            try:
                df = pd.read_parquet(filename)
                all_dfs.append(df)
                logger.info(f"Read {count} results from {filename}")
            except Exception as e:
                logger.exception(f"Error reading temporary file {filename}: {e}")
        if all_dfs:
            final_df = pd.concat(all_dfs, ignore_index=True)
            final_df.to_parquet(parquet_filename, index=False)
            logger.info(f"Successfully wrote {len(final_df)} total results to {parquet_filename}")
            for _, filename, _ in all_results:
                try:
                    os.remove(filename)
                except Exception:
                    pass
        else:
            logger.warning("No results to write to the final file!")
    except Exception as e:
        logger.exception(f"Error combining results: {e}")
    
    end_time_all = time.time()
    logger.info(f"All calculations completed in {end_time_all - start_time_all:.2f} seconds")

if __name__ == "__main__":
    main()