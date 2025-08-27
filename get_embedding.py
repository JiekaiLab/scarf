# CUDA_VISIBLE_DEVICES='4,5,6,7' WANDB_PROJECT="RNA_ATAC_FM [get_embeds]" python -m torch.distributed.run --nproc_per_node=4 --master_port=11140 get_embedding.py
import datetime
import pickle
import subprocess
import seaborn as sns
import numpy as np
import random
import torch
import pandas as pd
import json
from datasets import load_from_disk, concatenate_datasets, Dataset
from model import TotalModel_downstream, Pretrainer, TotalLossModel
from transformers.training_args import TrainingArguments
from transformers import BertConfig
from model.data_collator_modified import DataCollatorForLanguageModelingWithATAC_noSample
from model.pretrainer_modified import PreCollator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
from model.utils import load_prior_embedding, str2bool, load_model_with_index
import argparse
from tqdm import tqdm

os.environ["NCCL_DEBUG"] = "INFO"
os.environ["OMPI_MCA_opal_cuda_support"] = "true"
os.environ["CONDA_OVERRIDE_GLIBC"] = "2.56"

sns.set()
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_path", type=str,
                    default=None)
parser.add_argument("--dset_path", type=str, default="/media/GPU_Storage/zhaoyy/scFM_data_peak_filter/evaluation_data/PBMC/RNA_ATAC_data_v1")
parser.add_argument("--h5ad_dset_path", type=str, default=None)
parser.add_argument("--dset_name", type=str, default="PBMC")
parser.add_argument("--embed_type", type=str, default="RNA")
parser.add_argument("--has_label", type=str2bool, default=True)
parser.add_argument("--modality", type=int, default=0)
args = parser.parse_args()

checkpoint_path = args.checkpoint_path
import os

dset_name = args.dset_name

if dset_name in ['mKidney_10x_scM']:
    # List all items in the dataset directory
    dataset_items = os.listdir(args.dset_path)
    print(dataset_items)
    datasets = []
    
    for item in dataset_items:
        item_path = os.path.join(args.dset_path, item,'RNA_ATAC_data_v1')
        # Load each sub-dataset
        print(item_path)
        data_item = load_from_disk(item_path)
        datasets.append(data_item)
    
    data = concatenate_datasets(datasets) 
    

else:
    data = load_from_disk(args.dset_path)
data_modality = Dataset.from_dict({"modality": [args.modality] * data.num_rows})
dataset = concatenate_datasets([data, data_modality], axis=1)
print(dataset)
sorted_len = [32000] * dataset.num_rows
data_len = dataset.num_rows

dict_dir = '/public/home/t_zyy/scFM/shared/dict_data'
token_dict_path = f'{dict_dir}/hm_ENSG2token_dict.pickle'
with open(token_dict_path, "rb") as fp:
    token_dictionary = pickle.load(fp)


"""
set training parameters
"""
# total number of examples in Genecorpus-30M after QC filtering:
num_examples = dataset.num_rows
# number gpus
num_gpus = 8
# batch size for training and eval
# batch_size = 2 * 8
batch_size = 16
# max learning rate
max_lr = 1e-4
# learning schedule
lr_schedule_fn = "linear"
# warmup steps
warmup_steps_start = 0 #10_000
warmup_steps = 0#1_000
# number of epochs
epochs = 3
# optimizer
optimizer = "adamw_torch" #adamw_torch_fused
# weight_decay
weight_decay = 0.001
# rna_cluster labels and loss weights
bf16 = False
fp16 = False
fp16_opt_level = "O1"

# output directories
output_path = f'./get_embeds'
wandb_name = f"[get_embeds_{dset_name}]{checkpoint_path.split('/')[-2].split('__')[-1]}_{checkpoint_path.split('/')[-1]}_tvt"  # + f"_f{freeze_layers}"
# define output directory path
current_date = datetime.datetime.now()
datestamp = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}_{current_date.strftime('%X').replace(':', '')}"
output_dir = output_path + f"/{wandb_name.split(']')[1]}"
subprocess.call(f'mkdir {output_dir}', shell=True)


# prior knowledge
homologous_index, atac_embs = load_prior_embedding(dict_dir, len(token_dictionary))
priors = {}
priors['homologous_index'] = homologous_index
priors['hyenaDNA_embs'] = atac_embs
cell2cluster_dict = pd.read_pickle('/public/home/t_zyy/scFM/shared/dict_data/cell2cluster_dict_leiden_1.pickle')
priors['cell2cluster'] = cell2cluster_dict
priors['peak_idf'] = np.load('/public/home/t_zyy/scFM/shared/dict_data/peakToken_idf.npz')['arr_0']
priors['peak400_motif'] = torch.load("/public/home/t_zyy/scFM/shared/dict_data/hg38_peak400_motif_matrix.pt")


# load pretrained model
# loaded_state_dict = torch.load(checkpoint_path + '/pytorch_model.bin')
loaded_state_dict = load_model_with_index(checkpoint_path)
with open(checkpoint_path + '/config.json', 'r') as f:
    config = json.load(f)
config['modality_number'] = 2
config['rna_model_cfg']['use_rna_cl'] = False
config['atac_model_cfg']['use_atac_cl'] = False
loaded_config = BertConfig(**config)
model = TotalModel_downstream(loaded_config, priors=priors)
model.load_state_dict(loaded_state_dict, strict=False)
del loaded_state_dict
print(model)

# set training arguments
training_args = {
    "bf16": bf16,     # auto mixed precision
    "fp16": fp16,     # auto mixed precision
    "fp16_opt_level": fp16_opt_level,   # fp16 level
    "half_precision_backend": 'apex',
    "ddp_find_unused_parameters": True,
    "run_name": wandb_name,
    "dataloader_num_workers": num_gpus * 2,
    "dataloader_prefetch_factor": 2,
    "learning_rate": max_lr,
    "do_train": True,
    "do_eval": False,
    "group_by_length": False,
    "length_column_name": "length",
    "disable_tqdm": False,
    "lr_scheduler_type": lr_schedule_fn,
    "optim": optimizer,
    "adam_beta2": 0.999,
    "adam_epsilon": 1e-8,
    "warmup_steps": warmup_steps,
    "weight_decay": weight_decay,
    "per_device_train_batch_size": batch_size,
    "per_device_eval_batch_size": batch_size,
    "num_train_epochs": epochs,
    "save_total_limit": epochs,
    "save_strategy": "epoch",
    # "save_steps": np.floor(num_examples / batch_size / 8),  # 8 saves per epoch
    "logging_steps": 100,
    "output_dir": output_dir,
    "save_safetensors": False
}

training_args = TrainingArguments(**training_args)

# create the trainer
trainer = Pretrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset,
    example_lengths_file=sorted_len,
    token_dictionary=token_dictionary,
    data_collator=DataCollatorForLanguageModelingWithATAC_noSample(
        tokenizer=PreCollator(token_dictionary=token_dictionary),
        mlm=True, config=loaded_config, priors=priors,
    )
)

test_dataloader = trainer.get_test_dataloader(dataset)
trainer.model.eval()

is_only_emb = True
# direct_outs = [True,False]  #, False
direct_outs = [True]
for direct_out in direct_outs:
    rna_res, atac_res, cell_names, rna_kl_z, atac_kl_z = [], [], [], [], []
    for step, inputs in enumerate(tqdm(test_dataloader)):
        with torch.no_grad():
            inputs_new = {}
            if args.embed_type == 'RNA':
                to_cuda_names = ['rna_gene_ids', 'rna_gene_values', 'rna_attention_mask', 'species', 'modality']
            elif args.embed_type == 'ATAC':
                to_cuda_names = ['atac_cell_peaks', 'atac_lengths',
                                 'atac_peak_ids', 'atac_peak_idfs',
                                 'atac_attention_mask', 'species', 'modality']
            else:
                to_cuda_names = ['rna_gene_ids', 'rna_gene_values', 'rna_attention_mask',
                                 'atac_peak_ids', 'atac_peak_idfs',
                                 'atac_attention_mask', 'species', 'modality']
            #print(to_cuda_names)
            for each_name in to_cuda_names:
                if each_name in inputs:
                    inputs_new[each_name] = inputs[each_name].to("cuda")
            inputs_new['cell_name'] = inputs['cell_name']
            inputs_new['direct_out'] = direct_out
            if args.embed_type == 'RNA_and_ATAC':
                rna_res_dict, atac_res_dict, pred, gt, rna_kl_z_dict, atac_kl_z_dict = trainer.model.match_forward(**inputs_new)
                rna_res.extend(list(rna_res_dict.values()))
                atac_res.extend(list(atac_res_dict.values()))
                cell_names.extend(list(rna_res_dict.keys()))
                rna_kl_z.extend(list(rna_kl_z_dict.values()))
                atac_kl_z.extend(list(atac_kl_z_dict.values()))
                del rna_res_dict, atac_res_dict, pred, gt, rna_kl_z_dict, atac_kl_z_dict
            elif args.embed_type == 'RNA':
                rna_res_dict, rna_kl_z_dict = trainer.model.get_rna_embeddings(**inputs_new)
                rna_res.extend(list(rna_res_dict.values()))
                rna_kl_z.extend(list(rna_kl_z_dict.values()))
                cell_names.extend(list(rna_res_dict.keys()))
                del rna_res_dict, rna_kl_z_dict
            elif args.embed_type == 'ATAC':
                atac_res_dict, atac_kl_z_dict = trainer.model.get_atac_embeddings(**inputs_new)
                atac_res.extend(list(atac_res_dict.values()))
                atac_kl_z.extend(list(atac_kl_z_dict.values()))
                cell_names.extend(list(atac_res_dict.keys()))
                del atac_res_dict, atac_kl_z_dict
    save_path = f"{output_dir}/{dset_name}_{args.embed_type}_bs{batch_size}_directOut{int(direct_out)}"
    os.makedirs(save_path, exist_ok=True)

    if 'RNA' in args.embed_type:
        rna_res = rna_res[:data_len]
        cell_names = cell_names[:data_len]
        rna_kl_z = rna_kl_z[:data_len]
        print(data_len, np.array(rna_res).shape, len(cell_names))
        np.save(f'{save_path}/rna_cell_embs.npy', np.array(rna_res).astype(np.float32))
        np.save(f'{save_path}/cell_names.npy', np.array(cell_names))
        np.save(f'{save_path}/rna_kl_z.npy', np.array(rna_kl_z))
 #       os.system(f"python /public/home/t_lgl/ia_lgl/Ablation/utils/embedding_analysis.py \
 #        --raw {args.h5ad_dset_path} --embed_path {save_path} --type 'rna'")
 #       os.system(f"python /public/home/t_lgl/ia_lgl/Ablation/utils/embedding_analysis.py \
 #                --raw {args.h5ad_dset_path} --embed_path {save_path} --type 'rna' --name 'cosine'")
    if 'ATAC' in args.embed_type:
        atac_res = atac_res[:data_len]
        cell_names = cell_names[:data_len]
        atac_kl_z = atac_kl_z[:data_len]
        print(data_len, np.array(atac_res).shape, len(cell_names))
        np.save(f'{save_path}/atac_cell_embs.npy', np.array(atac_res).astype(np.float32))
        np.save(f'{save_path}/cell_names.npy', np.array(cell_names))
        np.save(f'{save_path}/atac_kl_z.npy', np.array(atac_kl_z))
 #       os.system(f"python /public/home/t_lgl/ia_lgl/Ablation/utils/embedding_analysis.py \
#         --raw {args.h5ad_dset_path} --embed_path {save_path} --type 'atac'")
#        os.system(f"python /public/home/t_lgl/ia_lgl/Ablation/utils/embedding_analysis.py \
#                 --raw {args.h5ad_dset_path} --embed_path {save_path} --type 'atac' --name 'cosine'")

    # get labels
    if args.has_label:
        cell_names = dataset['cell_name']
        cell_types = dataset['cell_types']
        name2type_dict = {cell_name: [cell_type] for cell_name, cell_type in zip(cell_names, cell_types)}
        name2type_df = pd.DataFrame(name2type_dict).T
        name2type_df.to_csv(f'{save_path}/labels.tsv.gz', header=False, index=True)
