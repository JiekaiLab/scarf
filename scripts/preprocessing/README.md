# Preprocessing Scripts

This folder archives standalone preprocessing scripts used to convert AnnData
inputs into HuggingFace `Dataset` objects for SCARF inference.

## Files

- `scM_convert.py`: standalone converter for `scRNA`, `scATAC`, and
  `scMultiome` inputs.

## Notes on `scM_convert.py`

`scM_convert.py` is the project preprocessing entrypoint for converting scRNA,
scATAC, and scMultiome AnnData files into SCARF-compatible datasets. It uses
in-file configuration values near the top of the script:

- `dict_data_dir`
- `sample_raw_names_all`
- `species_`
- `rna_raw_names_all`
- `atac_raw_names_all`
- `output_dir`
- `OMICS_TYPE`

It writes datasets under:

```text
processed_data/<sample_name>/data_<OMICS_TYPE>
```

The output is compatible with SCARF embedding inference when it contains the
fields used by `DataCollatorForLanguageModeling_Inference`, especially:

- RNA: `rna_gene_ids`, `rna_gene_values`, `rna_lengths`
- ATAC: `atac_cell_peaks`, `peak_num`
- shared: `species`, `cell_name`, optional `cell_types`, optional `batchs`

## Required Prior Files

Place the required prior files under the path configured by `dict_data_dir`.
The default value is `../dict_data`.

- `hm_ENSG2token_dict.pickle`
- `RNA_nonzero_median_10W.hg38.pickle` or
  `RNA_nonzero_median_10W.mm10.pickle`
- `peak2token_dict.pickle`

The optimized notebook in `downstream_tasks/preprocess.ipynb` additionally uses
`peakId2geneID_dict.pickle` and `ENSG2peakNum_dict.pickle` when generating
gene-level ATAC fields.

## Example

```bash
conda activate scarf
python scripts/preprocessing/scM_convert.py
```

For large datasets, monitor peak memory during preprocessing and write outputs
to a local high-throughput disk. The generated HuggingFace dataset can be loaded
by `downstream_tasks/embedding.ipynb` with `datasets.load_from_disk`.
