import tracemalloc
import time
tracemalloc.start()
start_time = time.time()

import os
import scanpy as sc
import anndata
import pandas as pd
import numpy as np
from typing import List, Dict
from tqdm import tqdm
from operator import itemgetter
from datasets import Dataset, Features, Sequence, Value, concatenate_datasets
import pickle
import multiprocessing


# meta =============================================================================================================
species2token = {'hg38': 0, 'mm10': 1}
dict_data_dir = '../dict_data'

sample_raw_names_all=['BMMC']
species_=['hg38']
rna_raw_names_all = [
    "../adata/10_hBMMC_10x/gex.h5ad"
]

atac_raw_names_all = [
    "../adata/10_hBMMC_10x/atac.h5ad"
]

# Define output directory
output_dir = "./processed_data/"
os.makedirs(output_dir, exist_ok=True)

# Cell filtering thresholds
MIN_ATAC_PEAKS = 1000
MIN_RNA_GENES = 50
CELL_BATCH_SIZE = 5000  # Number of cells per batch to control memory usage

# Metadata field names (customizable for different datasets)
CELL_TYPE_FIELD = 'cell_type'
BATCH_FIELD = 'batch'

# Omics type: scRNA, scATAC, scMultiome
OMICS_TYPE = 'scMultiome'


# RNA Data Processing and Ranking =================================================================================
def gene_filter_based_ENSid(ENSG2token: Dict[str, str], ENSG_list: List[str]) -> List[str]:
    '''
    Filter genes based on ENSEMBL ID to token mapping.
    Genes not in the mapping are marked as 'delete'.
    '''
    res = []
    for ENSG in ENSG_list:
        res.append(ENSG if ENSG in ENSG2token else "delete")
    return res

def Normalization_with_median(adata: anndata.AnnData, ENSid2median: Dict[str, int]) -> anndata.AnnData:
    '''
    Normalize expression data using gene-specific median values.
    '''
    if isinstance(adata.X, np.ndarray):
        X = adata.X
    else:
        X = adata.X.toarray()

    gene_nonzero_median = []
    for gene_ENSid in adata.var.gene_ids.to_list():
        gene_nonzero_median.append(ENSid2median.get(gene_ENSid, np.nan))
    gene_nonzero_median = np.array(gene_nonzero_median)

    adata.X = np.nan_to_num(X / np.tile(gene_nonzero_median, (X.shape[0], 1)))
    return adata

def rank_rna_value(adata: anndata.AnnData, ENSG2token: Dict[str, int], species='hg38'):
    '''
    Process RNA data: remove zeros, sort in descending order, and convert ENSEMBL IDs to tokens.
    Yields (cell_name, cell_data) one cell at a time to reduce memory usage.
    '''
    ENSid_list = adata.var.index.to_list()
    cell_names = adata.obs.index.to_list()

    for cell_idx, cell_data in enumerate(tqdm(adata.X)):
        nonzero_mask = np.nonzero(cell_data)[0]
        sorted_descend_indices = np.argsort(-cell_data[nonzero_mask])
        value = cell_data[nonzero_mask][sorted_descend_indices]
        ENSid_list_ = np.array(ENSid_list)[nonzero_mask][sorted_descend_indices]

        id_list = list(itemgetter(*ENSid_list_)(ENSG2token))

        cell_name = cell_names[cell_idx]
        cell_data_dict = {
            'input_ids': np.array(id_list).astype(np.int32),
            'values': np.array(value).astype(np.float32),
            'length': len(id_list),
            'species': species2token[species],
        }

        if CELL_TYPE_FIELD in adata.obs.keys():
            cell_data_dict['cell_types'] = adata.obs[CELL_TYPE_FIELD][cell_name]
        if BATCH_FIELD in adata.obs.keys():
            cell_data_dict['batchs'] = adata.obs[BATCH_FIELD][cell_name]

        yield cell_name, cell_data_dict

# ATAC Data Processing Functions ==============================================================================================
def peak_filter_based_name(peak2token: Dict[str, str], peak_name_list: List[str], species=None) -> List[str]:
    '''
    Filter peaks based on name to token mapping.
    Peaks not in the mapping are marked as 'delete'.
    '''
    res = []
    for peak_name in peak_name_list:
        res.append(f"{species}_{peak_name}" if f"{species}_{peak_name}" in peak2token else "delete")
    return res

def rank_atac_peaks(adata: anndata.AnnData, peak2token: Dict[str, int], species):
    '''
    Process ATAC data: remove zeros and convert peak names to tokens.
    Yields (cell_name, cell_data) one cell at a time to reduce memory usage.
    '''
    peak_name_list = adata.var.index.to_list()  # already in f"{species}_{peak_name}" format after filtering
    peak_token_list = [peak2token.get(p) for p in peak_name_list]  # key already includes species prefix
    cell_names = adata.obs.index.to_list()

    for cell_idx, cell_data in enumerate(tqdm(adata.X)):
        if not isinstance(cell_data, np.ndarray):
            cell_data = cell_data.toarray().flatten()
        nonzero_mask = np.nonzero(cell_data)[0]
        peak_token_list_cell = np.array(peak_token_list)[nonzero_mask]
        cell_name = cell_names[cell_idx]
        cell_data_dict = {
            'peaks': peak_token_list_cell,
            'species': species2token[species],
        }

        if CELL_TYPE_FIELD in adata.obs.keys():
            cell_data_dict['cell_types'] = adata.obs[CELL_TYPE_FIELD][cell_name]
        if BATCH_FIELD in adata.obs.keys():
            cell_data_dict['batchs'] = adata.obs[BATCH_FIELD][cell_name]

        yield cell_name, cell_data_dict


# Cell Processing Functions ====================================================================================================
def process_rna(each_cell_name):
    """Process RNA data for a single cell (scRNA or scMultiome mode)."""
    rna_data = rna_cell2data[each_cell_name]
    rna_gene_ids = rna_data['input_ids']
    rna_gene_values = rna_data['values']
    rna_length = rna_data['length']
    cell_type = rna_data['cell_types'] if 'cell_types' in rna_data else None
    batch = rna_data['batchs'] if 'batchs' in rna_data else None

    if rna_length < MIN_RNA_GENES:
        return []

    return (rna_gene_ids, rna_gene_values, rna_length, each_cell_name, cell_type, batch)


def process_atac(each_cell_name):
    """Process ATAC data for a single cell (scATAC or scMultiome mode)."""
    cell_data = atac_cell2data[each_cell_name]
    atac_data = cell_data['peaks']
    peak_num = len(atac_data)
    cell_type = cell_data.get('cell_types', None)
    batch = cell_data.get('batchs', None)

    if len(atac_data) < MIN_ATAC_PEAKS:
        return []

    return (atac_data, peak_num, each_cell_name, cell_type, batch)


def process_multiome(each_cell_name):
    """Process multi-omics data for a single cell."""
    rna_data = rna_cell2data[each_cell_name]
    atac_cell_data = atac_cell2data[each_cell_name]
    atac_data = atac_cell_data['peaks']
    peak_num = len(atac_data)

    if len(atac_data) < MIN_ATAC_PEAKS:
        return []

    rna_gene_ids = rna_data['input_ids']
    rna_gene_values = rna_data['values']
    rna_length = rna_data['length']
    cell_type = rna_data['cell_types'] if 'cell_types' in rna_data else None
    batch = rna_data['batchs'] if 'batchs' in rna_data else None

    if rna_length < MIN_RNA_GENES:
        return []

    return (rna_gene_ids, rna_gene_values, rna_length, peak_num, each_cell_name, atac_data, cell_type, batch)


# Save Data Function ====================================================================================================
def save_data(path: str, dataset: Dataset, rna_length: List[int] = None, peak_nums: List[int] = None) -> None:
    '''save dataset to path'''
    dataset.save_to_disk(path)
    if rna_length is not None:
        sorted_rna = sorted(rna_length)
        with open(path + '/sorted_rna_length.pickle', 'wb') as f:
            pickle.dump(sorted_rna, f)
    if peak_nums is not None:
        sorted_atac = sorted(peak_nums)
        with open(path + '/sorted_atac_length.pickle', 'wb') as f:
            pickle.dump(sorted_atac, f)


# Main Processing Loop ====================================================================================================
for sample_raw_name, species, sample_file_name_rna, sample_file_name_atac in zip(
    sample_raw_names_all, species_, rna_raw_names_all, atac_raw_names_all):

    save_path = output_dir + sample_raw_name + f'/data_{OMICS_TYPE}'
    print(f'Processing sample: {sample_raw_name} ({OMICS_TYPE} mode)')

    # Initialize variables
    rna_cell2data = {}
    atac_cell2data = {}

    # ---- scRNA or scMultiome: Load and process RNA data ----
    if OMICS_TYPE in ['scRNA', 'scMultiome']:
        print("1. Loading ENSG2token dictionary")
        ENSG2token_path = f'{dict_data_dir}/hm_ENSG2token_dict.pickle'
        ENSG2token = pd.read_pickle(ENSG2token_path)

        print("2. Loading median expression dictionary")
        ENSG2median_path = f'{dict_data_dir}/RNA_nonzero_median_10W.{species}.pickle'
        ENSG2median = pd.read_pickle(ENSG2median_path)

        print("3. Loading RNA data")
        adata_rna_ = sc.read(sample_file_name_rna)
        adata_rna = anndata.AnnData(X=adata_rna_.X, var=adata_rna_.var, obs=adata_rna_.obs)

        print("4. Filtering genes")
        gene_ENSid_list = gene_filter_based_ENSid(ENSG2token, adata_rna.var.gene_ids.to_list())
        adata_rna.var['gene_names'] = adata_rna.var.index.tolist()
        adata_rna.var.index = gene_ENSid_list
        adata_rna = adata_rna[:, adata_rna.var.index != "delete"]

        print("5. Normalizing and processing RNA data")
        sc.pp.normalize_total(adata_rna, target_sum=1e4, inplace=True)
        sc.pp.log1p(adata_rna)
        adata_rna = Normalization_with_median(adata_rna, ENSG2median)

        print("5b. Building RNA cell2data dict")
        for cell_name, cell_data in rank_rna_value(adata_rna, ENSG2token, species=species):
            rna_cell2data[cell_name] = cell_data

    # ---- scATAC or scMultiome: Load and process ATAC data ----
    if OMICS_TYPE in ['scATAC', 'scMultiome']:
        print("6. Loading peak dictionaries")
        peak2token_path = f'{dict_data_dir}/peak2token_dict.pickle'
        peak2token = pd.read_pickle(peak2token_path)

        print("7. Loading ATAC data")
        adata_atac = sc.read(sample_file_name_atac)

        print("8. Filtering ATAC peaks")
        peak_name_list = peak_filter_based_name(peak2token, adata_atac.var.index.to_list(), species=species)
        adata_atac.var.index = peak_name_list
        adata_atac = adata_atac[:, adata_atac.var.index != "delete"]

        print("9. Ranking ATAC peaks")
        for cell_name, cell_data in rank_atac_peaks(adata_atac, peak2token, species=species):
            atac_cell2data[cell_name] = cell_data

    # ---- Determine cell names based on omics type ----
    if OMICS_TYPE == 'scRNA':
        cell_names = list(rna_cell2data.keys())
    elif OMICS_TYPE == 'scATAC':
        cell_names = list(atac_cell2data.keys())
    else:  # scMultiome
        rna_cell_names = list(rna_cell2data.keys())
        atac_cell_names = list(atac_cell2data.keys())
        cell_names = sorted(list(set(rna_cell_names) & set(atac_cell_names)))

    print(f"Sample {sample_raw_name} has {len(cell_names)} cells")

    # ---- Initialize lists for processed data ----
    rna_gene_ids = []
    rna_gene_values = []
    rna_lengths = []
    cell_names_processed = []
    peak_nums = []
    atac_cell_peaks = []
    cell_types = []
    batchs = []

    # ---- Process cells based on omics type ----
    total_cells = len(cell_names)
    num_batches = (total_cells + CELL_BATCH_SIZE - 1) // CELL_BATCH_SIZE

    # Accumulate all batches
    all_rna_gene_ids = []
    all_rna_gene_values = []
    all_rna_lengths = []
    all_cell_names_processed = []
    all_peak_nums = []
    all_atac_cell_peaks = []
    all_cell_types = []
    all_batchs = []

    for batch_idx in range(num_batches):
        batch_start = batch_idx * CELL_BATCH_SIZE
        batch_end = min(batch_start + CELL_BATCH_SIZE, total_cells)
        batch_cell_names = cell_names[batch_start:batch_end]

        print(f"  Batch {batch_idx+1}/{num_batches}: processing cells {batch_start}-{batch_end}")

        # Initialize batch lists
        batch_rna_gene_ids = []
        batch_rna_gene_values = []
        batch_rna_lengths = []
        batch_cell_names_processed = []
        batch_peak_nums = []
        batch_atac_cell_peaks = []
        batch_cell_types = []
        batch_batchs = []

        # ---- Parallel processing within batch ----
        pool = multiprocessing.Pool(processes=2)

        if OMICS_TYPE == 'scRNA':
            results = pool.imap(process_rna, batch_cell_names)
            for res in results:
                if res:
                    batch_rna_gene_ids.append(res[0])
                    batch_rna_gene_values.append(res[1])
                    batch_rna_lengths.append(res[2])
                    batch_cell_names_processed.append(res[3])
                    batch_cell_types.append(res[4])
                    batch_batchs.append(res[5])

        elif OMICS_TYPE == 'scATAC':
            results = pool.imap(process_atac, batch_cell_names)
            for res in results:
                if res:
                    batch_atac_cell_peaks.append(res[0])
                    batch_peak_nums.append(res[1])
                    batch_cell_names_processed.append(res[2])
                    batch_cell_types.append(res[3])
                    batch_batchs.append(res[4])

        else:  # scMultiome
            results = pool.imap(process_multiome, batch_cell_names)
            for res in results:
                if res:
                    batch_rna_gene_ids.append(res[0])
                    batch_rna_gene_values.append(res[1])
                    batch_rna_lengths.append(res[2])
                    batch_peak_nums.append(res[3])
                    batch_cell_names_processed.append(res[4])
                    batch_atac_cell_peaks.append(res[5])
                    batch_cell_types.append(res[6])
                    batch_batchs.append(res[7])

        pool.close()
        pool.join()

        print(f"  Batch {batch_idx+1}/{num_batches}: {len(batch_cell_names_processed)} cells passed filter")

        # Accumulate batch results
        all_rna_gene_ids.extend(batch_rna_gene_ids)
        all_rna_gene_values.extend(batch_rna_gene_values)
        all_rna_lengths.extend(batch_rna_lengths)
        all_cell_names_processed.extend(batch_cell_names_processed)
        all_peak_nums.extend(batch_peak_nums)
        all_atac_cell_peaks.extend(batch_atac_cell_peaks)
        all_cell_types.extend(batch_cell_types)
        all_batchs.extend(batch_batchs)

        del batch_rna_gene_ids, batch_rna_gene_values
        del batch_rna_lengths, batch_peak_nums, batch_atac_cell_peaks
        del batch_cell_types, batch_batchs, batch_cell_names_processed

    print(f"Sample {sample_raw_name}: {len(all_cell_names_processed)} cells processed in {num_batches} batches")

    # ---- Build final data_dict and structure ----
    if OMICS_TYPE == 'scRNA':
        data_dict = {
            'rna_gene_ids': all_rna_gene_ids,
            'rna_gene_values': all_rna_gene_values,
            'rna_lengths': all_rna_lengths,
            'species': [species2token[species]] * len(all_cell_names_processed),
            'cell_name': all_cell_names_processed,
        }
        structure = Features({
            'rna_gene_ids': Sequence(feature=Value(dtype='int32')),
            'rna_gene_values': Sequence(feature=Value(dtype='float32')),
            'rna_lengths': Value(dtype='int16'),
            'species': Value(dtype='int8'),
            'cell_name': Value(dtype='string'),
        })

    elif OMICS_TYPE == 'scATAC':
        data_dict = {
            'atac_cell_peaks': all_atac_cell_peaks,
            'peak_num': all_peak_nums,
            'species': [species2token[species]] * len(all_cell_names_processed),
            'cell_name': all_cell_names_processed,
        }
        structure = Features({
            'atac_cell_peaks': Sequence(feature=Value(dtype='int32')),
            'peak_num': Value(dtype='int32'),
            'species': Value(dtype='int8'),
            'cell_name': Value(dtype='string'),
        })

    else:  # scMultiome
        data_dict = {
            'rna_gene_ids': all_rna_gene_ids,
            'rna_gene_values': all_rna_gene_values,
            'rna_lengths': all_rna_lengths,
            'atac_cell_peaks': all_atac_cell_peaks,
            'peak_num': all_peak_nums,
            'species': [species2token[species]] * len(all_cell_names_processed),
            'cell_name': all_cell_names_processed,
        }
        structure = Features({
            'rna_gene_ids': Sequence(feature=Value(dtype='int32')),
            'rna_gene_values': Sequence(feature=Value(dtype='float32')),
            'rna_lengths': Value(dtype='int16'),
            'atac_cell_peaks': Sequence(feature=Value(dtype='int32')),
            'peak_num': Value(dtype='int32'),
            'species': Value(dtype='int8'),
            'cell_name': Value(dtype='string'),
        })

    # ---- Add optional fields ----
    if None not in all_batchs:
        data_dict['batchs'] = all_batchs
        structure = Features(structure)
        structure['batchs'] = Value(dtype='string')

    if None not in all_cell_types:
        data_dict['cell_types'] = all_cell_types
        structure = Features(structure)
        structure['cell_types'] = Value(dtype='string')

    # ---- Save final dataset ----
    dataset = Dataset.from_dict(data_dict, features=structure)
    os.makedirs(save_path, exist_ok=True)
    save_data(save_path, dataset,
              rna_length=all_rna_lengths if OMICS_TYPE != 'scATAC' else None,
              peak_nums=all_peak_nums if OMICS_TYPE != 'scRNA' else None)

    del data_dict
    print(f"Completed processing for sample: {sample_raw_name}")


end_time = time.time()
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

print(f"\n=== Performance Stats ===")
print(f"Execution time: {end_time - start_time:.2f} seconds")
print(f"Current memory: {current / 1024 / 1024:.2f} MB")
print(f"Peak memory: {peak / 1024 / 1024:.2f} MB")
