from collections.abc import Mapping
from typing import List, Union, Dict, Any, Tuple, Optional
from transformers.data.data_collator import _torch_collate_batch
import torch
from transformers import DataCollatorForLanguageModeling
import numpy as np
import random

random.seed(42)

class DataCollatorForLanguageModeling_Inference(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, mlm, config, priors, **kwargs) -> None:
        super().__init__(tokenizer, mlm)
        if 'RNA' in config.pretrain_mode:
            self.rna_max_length = config.rna_model_cfg['rna_max_input_size']
            self.rna_mlm_probability = config.rna_model_cfg['rna_mlm_probability']

        if 'ATAC' in config.pretrain_mode:
            self.atac_max_length = config.atac_model_cfg['atac_max_input_size']
            self.atac_mlm_probability = config.atac_model_cfg['atac_mlm_probability']
            self.x_peak_inacc_ratio = config.atac_model_cfg['x_peak_inacc_ratio']

        self.config = config
        self.peak_idf = priors['peak_idf']

    def _pad(self, input, max_length):
        length = len(input)
        if length >= max_length:
            return input[:max_length]
        difference = max_length - length
        return input + [self.tokenizer.pad_token_id] * difference

    def _pad_atac(self, input, max_length):
        length = len(input)
        if length >= max_length:
            return input[:max_length], np.ones(max_length)
        difference = max_length - length
        return input.tolist() + [self.tokenizer.pad_token_id] * difference, np.array([1] * len(input) +[-100] * difference)

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], Mapping):
            batch = {}
            if 'rna_gene_ids' in examples[0] and 'RNA' in self.config.pretrain_mode:
                input_ids = [self._pad(example['rna_gene_ids'], max_length=self.rna_max_length) for example in examples]
                rna_lenths = torch.tensor([example['rna_lengths'] for example in examples])
                batch['rna_gene_ids'] = torch.tensor(input_ids)

                values = [self._pad(example['rna_gene_values'], max_length=self.rna_max_length) for example in examples]
                batch['rna_gene_values'] = torch.tensor(values)

                batch['rna_lengths'] = rna_lenths
                rna_attention_mask = torch.zeros_like(batch['rna_gene_ids'])
                for i, length in enumerate(rna_lenths):
                    rna_attention_mask[i, :length] = 1
                batch['rna_attention_mask'] = rna_attention_mask


            if 'atac_cell_peaks' in examples[0] and 'ATAC' in self.config.pretrain_mode:
                atac_peak_idfs, atac_peak_ids, ranked_peak_acc_labels = [], [], []

                for example in examples:
                    cell_peak_idf = self.peak_idf[example['atac_cell_peaks']]
                    arg_sort = np.argsort(-cell_peak_idf)
                    ranked_peak_idf, ranked_peak_acc_label = self._pad_atac(cell_peak_idf[arg_sort], max_length=self.atac_max_length)
                    ranked_peak_idf = np.array(ranked_peak_idf) / example['peak_num'] * 1000
                    ranked_peak_id = np.array(self._pad(np.array(example['atac_cell_peaks'])[arg_sort].tolist(), max_length=self.atac_max_length))
                    atac_peak_idfs.append(ranked_peak_idf)
                    atac_peak_ids.append(ranked_peak_id)
                    ranked_peak_acc_labels.append(ranked_peak_acc_label)

                atac_peak_idfs = torch.tensor(np.array(atac_peak_idfs))
                cell_peak_nums = torch.tensor([example['peak_num'] for example in examples])
                atac_attention_mask = torch.zeros_like(atac_peak_idfs)
                for i, length in enumerate(cell_peak_nums):
                    atac_attention_mask[i, :length] = 1

                batch.update({
                    'atac_peak_ids': torch.tensor(np.array(atac_peak_ids)),
                    'atac_peak_idfs': atac_peak_idfs,
                    'atac_peak_acc_labels': torch.tensor(np.array(ranked_peak_acc_labels)),
                    'atac_attention_mask': atac_attention_mask,
                })

            if 'species' in examples[0]:
                species = torch.tensor([example['species'] for example in examples])
                batch['species'] = species
            if 'modality' in examples[0]:
                modality = torch.tensor([example['modality'] for example in examples])
                batch['modality'] = modality
            if 'cell_types' in examples[0]:
                cell_types = [example['cell_types'] for example in examples]
                batch['cell_types'] = cell_types
            if 'cell_name' in examples[0]:
                cell_name = [example['cell_name'] for example in examples]
                batch['cell_name'] = cell_name
            if 'labels' in examples[0]:
                label = torch.tensor([example['labels'] for example in examples])
                batch['labels'] = label
        else:
            batch = {
                "rna_gene_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }

        return batch