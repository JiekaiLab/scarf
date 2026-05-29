# Prior Data

This folder stores the token dictionaries and prior statistics used by SCARF
preprocessing and embedding inference. Large prior files are distributed through
Zenodo instead of being committed to the repository.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17205044.svg)](https://doi.org/10.5281/zenodo.17205044)

Download the prior data archive from Zenodo and place the extracted files in
this folder.

## File Reference

| File | Used by | Required when |
| --- | --- | --- |
| `hm_ENSG2token_dict.pickle` | preprocessing, embedding | Tokenizing RNA genes and running RNA/multiome inference. |
| `peak2token_dict.pickle` | preprocessing | Tokenizing ATAC peaks from raw AnnData inputs. |
| `peakId2geneID_dict.pickle` | optimized preprocessing notebook | Building gene-level ATAC fields from peak-to-gene prior links. |
| `ENSG2peakNum_dict.pickle` | optimized preprocessing notebook | Normalizing gene-level ATAC peak counts. |
| `RNA_nonzero_median_10W.hg38.pickle` | preprocessing | Normalizing human RNA inputs. |
| `RNA_nonzero_median_10W.mm10.pickle` | preprocessing | Normalizing mouse RNA inputs. |
| `peakToken_idf.npz` | embedding collator | Running ATAC or multiome embedding inference. |

## Minimal Requirements

- RNA-only preprocessing: `hm_ENSG2token_dict.pickle` plus the matching
  `RNA_nonzero_median_10W.<species>.pickle`.
- ATAC-only preprocessing: `peak2token_dict.pickle`.
- Multiome preprocessing with gene-level ATAC fields:
  `hm_ENSG2token_dict.pickle`, `peak2token_dict.pickle`,
  `peakId2geneID_dict.pickle`, `ENSG2peakNum_dict.pickle`, and the matching
  RNA median file.
- Embedding inference from an already preprocessed dataset:
  `hm_ENSG2token_dict.pickle` for RNA inputs and `peakToken_idf.npz` for ATAC
  inputs.
