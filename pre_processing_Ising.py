import os
import sys
import time
import logging
import random
import warnings
import gc
import psutil
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from datetime import datetime
import cupy as cp
import cupyx.scipy.sparse as sparse
from cupyx.scipy.sparse import linalg as splinalg
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from torch_geometric.data import InMemoryDataset, Data
from itertools import combinations


def get_available_gpus():
    try:
        n_gpus = cp.cuda.runtime.getDeviceCount()
        logging.info(f"Found {n_gpus} GPUs")
        return n_gpus
    except Exception as e:
        logging.error(f"Error detecting GPUs: {str(e)}")
        return 1  


CONFIG = {
    'data_paths': [
        'transverse_ising_data_large_4k.parquet',
        'transverse_ising_data_large_10k.parquet',
    ],
    'processed_dir': './processed_Ising_full_large',
    'processed_file_name': 'data.pt',
    'random_seed': 42,
    'chunk_size': 400,  
    'save_interval': 150,  
    'num_workers': min(mp.cpu_count()-1, 40),  
    'batch_size': 5,  
    'timeout': 600,  
    'num_gpus': 1, 
    'max_points_per_ns': {  
        2: 3000,   
        3: 5000,
        4: 10000,
        5: 20000,
        6: 30000,
        7: 40000,
        8: 50000,
        9: 60000,
        10: 70000,
        11: 80000,
        12: 90000,
        13: 500000,
        14: 500000,
        15: 500000,
        16: 500000,
        17: 500000,
        18: 500000,
        19: 500000,
        20: 500000,  
    }
}

def setup_logging():
    os.makedirs(CONFIG['processed_dir'], exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(CONFIG['processed_dir'], 'processing.log'))
        ]
    )

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cp.random.seed(seed)

def set_gpu(gpu_id):
    try:
        cp.cuda.Device(gpu_id).use()
        logging.info(f"Using GPU device {gpu_id}")
    except Exception as e:
        logging.error(f"Error setting GPU device {gpu_id}: {str(e)}")


def calculate_quantum_correlations_optimized(state_indices, state_probs, N):
    states = cp.array(state_indices, dtype=cp.int64)
    probs = cp.array(state_probs, dtype=cp.float32)
    bit_positions = cp.arange(N, dtype=cp.int64).reshape(-1, 1)
    bit_masks = (states & (1 << bit_positions)) != 0
    diag_correlations = cp.dot(bit_masks, probs)
    return diag_correlations

def create_edges(positions, N):
    i_indices, j_indices = cp.triu_indices(N, k=1)
    edges = cp.vstack((i_indices, j_indices))
    return edges if edges.size > 0 else cp.zeros((2, 0), dtype=cp.int64)

def process_single_row(row_data, gpu_id=0):
    try:
        set_gpu(gpu_id % CONFIG['num_gpus'])
        N = row_data['N_s']
        positions = cp.arange(N, dtype=cp.float32)
        spins_probs = calculate_quantum_correlations_optimized(
            row_data['All_Indices'], row_data['All_Probabilities'], N)
        mask = cp.array([int(bit) for bit in row_data['Subsystem_Mask']], dtype=cp.float32).reshape(-1, 1)
        node_features = cp.concatenate([
            positions.reshape(-1, 1),
            spins_probs.reshape(-1, 1),
            mask
        ], axis=1)
        edge_index = create_edges(positions, N)
        if edge_index.size > 0:
            pos_i = positions[edge_index[0]]
            pos_j = positions[edge_index[1]]
            distance = pos_j - pos_i
            dist_ij = distance.reshape(-1, 1) / N
            states = cp.array(row_data['All_Indices'], dtype=cp.int64)
            probs = cp.array(row_data['All_Probabilities'], dtype=cp.float32)
            edge_corr_values = []
            for i, j in zip(edge_index[0].get(), edge_index[1].get()):
                mask_i = (states & (1 << i)) != 0
                mask_j = (states & (1 << j)) != 0
                mask_ij = mask_i & mask_j
                corr_ij = cp.sum(probs[mask_ij])
                connected_corr = corr_ij - spins_probs[i] * spins_probs[j]
                edge_corr_values.append(connected_corr)
            edge_corr_values = cp.array(edge_corr_values, dtype=cp.float32).reshape(-1, 1)
            edge_attr = cp.concatenate([edge_corr_values, dist_ij], axis=1)
        else:
            edge_attr = cp.zeros((0, 2), dtype=cp.float32)
        return {
            'node_features': node_features,
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'target_vne': cp.array([row_data['Von_Neumann_Entropy']], dtype=cp.float32),
            'system_size': cp.array([[N]], dtype=cp.float32),
            'nA': cp.array([[float(mask.sum())]], dtype=cp.float32),
            'nB': cp.array([[float(N - mask.sum())]], dtype=cp.float32),
            'J_over_h': cp.array([row_data['J_over_h']], dtype=cp.float32)
        }
    except Exception as e:
        logging.error(f"Error processing row on GPU {gpu_id}: {str(e)}")
        return None

def process_chunk(chunk_data, worker_id=0):
    results = []
    gpu_id = worker_id % CONFIG['num_gpus']
    for _, row in chunk_data.iterrows():
        data = process_single_row(row, gpu_id)
        if data is not None:
            results.append(data)
    return results

class SpinDataset(InMemoryDataset):
    def __init__(self, dataframe, root='.', transform=None, pre_transform=None):
        self.df = dataframe
        if 'column_names' not in CONFIG:
            CONFIG['column_names'] = dataframe.columns.tolist()
        super().__init__(root, transform, pre_transform)
        if os.path.exists(self.processed_paths[0]):
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            self.data, self.slices = self.process()

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [CONFIG['processed_file_name']]

    def download(self):
        pass

    def cupy_to_torch_data(self, cupy_data, device='cpu'):
        try:
            data = Data(
                x=torch.from_numpy(cp.asnumpy(cupy_data['node_features'])).float(),
                edge_index=torch.from_numpy(cp.asnumpy(cupy_data['edge_index'])).long(),
                edge_attr=torch.from_numpy(cp.asnumpy(cupy_data['edge_attr'])).float(),
                y=torch.from_numpy(cp.asnumpy(cupy_data['target_vne'])).float(),
                system_size=torch.from_numpy(cp.asnumpy(cupy_data['system_size'])).float(),
                nA=torch.from_numpy(cp.asnumpy(cupy_data['nA'])).float(),
                nB=torch.from_numpy(cp.asnumpy(cupy_data['nB'])).float()
            )
            return data.to(device) if device != 'cpu' else data
        except Exception as e:
            logging.error(f"Error converting cupy to torch: {str(e)}")
            return None

    def process(self):
        try:
            filtered_data = []
            for ns, max_points in CONFIG['max_points_per_ns'].items():
                ns_data = self.df[self.df['N_s'] == ns]
                if len(ns_data) > max_points:
                    ns_data = ns_data.sample(max_points, random_state=CONFIG['random_seed'])
                filtered_data.append(ns_data)
            self.df = pd.concat(filtered_data, ignore_index=True)
            self.df = self.df.sample(frac=1, random_state=CONFIG['random_seed']).reset_index(drop=True)
            num_chunks = len(self.df) // CONFIG['chunk_size'] + 1
            chunks = np.array_split(self.df, num_chunks)
            processed_chunks = []
            total_processed = 0
            start_time = time.time()
            last_save = start_time
            logging.info(f"Starting parallel processing with {len(chunks)} chunks using {CONFIG['num_gpus']} GPUs")
            for batch_idx in range(0, len(chunks), CONFIG['batch_size']):
                batch_start = time.time()
                batch_chunks = chunks[batch_idx : batch_idx + CONFIG['batch_size']]
                logging.info(f"\nStarting batch {batch_idx // CONFIG['batch_size'] + 1}/{(len(chunks)-1) // CONFIG['batch_size'] + 1}")
                logging.info(f"Total rows processed so far: {total_processed}")
                with ProcessPoolExecutor(max_workers=CONFIG['num_workers']) as executor:
                    future_to_chunk = {
                        executor.submit(process_chunk, chunk, i): i 
                        for i, chunk in enumerate(batch_chunks)
                    }
                    for future in as_completed(future_to_chunk):
                        chunk_idx = batch_idx + future_to_chunk[future]
                        try:
                            chunk_results = future.result(timeout=CONFIG['timeout'])
                            if chunk_results:
                                processed_chunks.extend(chunk_results)
                                total_processed += len(chunk_results)
                                logging.info(
                                    f"Chunk {chunk_idx+1}/{len(chunks)} complete. Processed {len(chunk_results)} rows (Total: {total_processed})"
                                )
                                current_time = time.time()
                                if (current_time - last_save > 600 or ((chunk_idx+1) % CONFIG['save_interval'] == 0)):
                                    temp_save_path = os.path.join(
                                        CONFIG['processed_dir'], 
                                        f'temp_cupy_chunk_{chunk_idx+1}.npy'
                                    )
                                    np.save(temp_save_path, [cp.asnumpy(chunk) if isinstance(chunk, cp.ndarray) else chunk for chunk in processed_chunks])
                                    memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
                                    logging.info(f"Saved intermediate results at chunk {chunk_idx+1}. Memory usage: {memory_usage:.2f} MB")
                                    last_save = current_time
                        except TimeoutError:
                            logging.error(f"Timeout processing chunk {chunk_idx+1}")
                            continue
                        except Exception as e:
                            logging.error(f"Error processing chunk {chunk_idx+1}: {str(e)}")
                            continue
                batch_time = time.time() - batch_start
                logging.info(f"Batch complete. Time: {batch_time:.2f}s. Average per chunk: {batch_time/len(batch_chunks):.2f}s")
                gc.collect()
                for device_id in range(CONFIG['num_gpus']):
                    try:
                        with cp.cuda.Device(device_id):
                            cp.get_default_memory_pool().free_all_blocks()
                            logging.info(f"Cleared memory for GPU {device_id}")
                    except Exception as e:
                        logging.warning(f"Error clearing memory for GPU {device_id}: {str(e)}")
            logging.info("Processing complete. Converting to PyTorch Geometric format...")
            if not processed_chunks:
                raise RuntimeError("No data was successfully processed")
            data_list = []
            for cupy_data in processed_chunks:
                torch_data = self.cupy_to_torch_data(cupy_data)
                if torch_data is not None:
                    data_list.append(torch_data)
            for temp_file in os.listdir(CONFIG['processed_dir']):
                if temp_file.startswith('temp_cupy_chunk_'):
                    try:
                        os.remove(os.path.join(CONFIG['processed_dir'], temp_file))
                    except Exception as e:
                        logging.warning(f"Failed to remove temporary file {temp_file}: {str(e)}")
            data, slices = self.collate(data_list)
            torch.save((data, slices), os.path.join(CONFIG['processed_dir'], CONFIG['processed_file_name']))
            total_time = time.time() - start_time
            logging.info(f"Total processing time: {total_time:.2f}s")
            logging.info(f"Processed {total_processed} rows successfully")
            return data, slices
        except Exception as e:
            logging.error(f"Error in process method: {str(e)}")
            raise

def load_data():
    df_list = []
    for path in CONFIG['data_paths']:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found at {path}")
        logging.info(f"Loading data from {path}")
        table = pq.read_table(path)
        df_temp = table.to_pandas()
        df_list.append(df_temp)
    df = pd.concat(df_list, ignore_index=True)
    filtered_data = []
    for ns, max_points in CONFIG['max_points_per_ns'].items():
        ns_data = df[df['N_s'] == ns]
        if len(ns_data) > max_points:
            ns_data = ns_data.sample(max_points, random_state=CONFIG['random_seed'])
        filtered_data.append(ns_data)
    df_filtered = pd.concat(filtered_data, ignore_index=True)
    df_shuffled = df_filtered.sample(frac=1, random_state=CONFIG['random_seed']).reset_index(drop=True)
    return SpinDataset(dataframe=df_shuffled, root=CONFIG['processed_dir'])

def main():
    os.makedirs(CONFIG['processed_dir'], exist_ok=True)
    setup_logging()
    set_seed(CONFIG['random_seed'])
    logging.info(f"Processing with {CONFIG['num_gpus']} GPUs in parallel")
    for gpu_id in range(CONFIG['num_gpus']):
        try:
            with cp.cuda.Device(gpu_id):
                test_array = cp.array([1, 2, 3])
                logging.info(f"Successfully initialized GPU {gpu_id}")
        except Exception as e:
            logging.error(f"Failed to initialize GPU {gpu_id}: {str(e)}")
    torch.set_num_threads(CONFIG['num_workers'])
    try:
        dataset = load_data()
        logging.info(f"Finished processing. Dataset length: {len(dataset)}")
        if len(dataset) > 0:
            logging.info(f"Sample data object: {dataset[0]}")
            logging.info(f"Node feature dimensions: {dataset[0].x.shape}")
            logging.info("Node features include: positions (1), spin prob (1), subsystem mask (1)")
            logging.info(f"This is a 1D chain with {dataset[0].system_size.item()} nodes")
    except Exception as e:
        logging.error(f"Failed to process dataset: {str(e)}")
        raise

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()