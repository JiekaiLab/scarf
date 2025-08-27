from typing import Optional, Tuple, Union
from .modeling_bert_modified import BertForMaskedLMWithRNA, BertForMaskedLMWithATAC, BertPreTrainedModel
from collections import OrderedDict
import torch
from torch import nn
import numpy as np


class TotalLossModel(BertPreTrainedModel):
    def __init__(self, config, priors):
        super(TotalLossModel, self).__init__(config)
        self.config = config
        if 'RNA' in config.pretrain_mode:
            self.encoder_rna = BertForMaskedLMWithRNA(config, priors)
            self.logit_scale_rna_R1R2 = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).float()
            self.rna_cell_projection = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(config.rna_model_cfg['hidden_size'], config.mlp_out_size)),
                ("ln", nn.LayerNorm(config.mlp_out_size))
            ]))
            self.rna_gene_projection = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(config.rna_model_cfg['hidden_size'], config.mlp_out_size)),
                ("ln", nn.LayerNorm(config.mlp_out_size))
            ]))

        if 'ATAC' in config.pretrain_mode:
            self.encoder_atac = BertForMaskedLMWithATAC(config, priors)
            self.logit_scale_atac_A1A2 = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).float()
            self.atac_cell_projection = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(config.atac_model_cfg['hidden_size'], config.mlp_out_size)),
                ("ln", nn.LayerNorm(config.mlp_out_size))
            ]))
            self.atac_gene_projection = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(config.atac_model_cfg['hidden_size'], config.mlp_out_size)),
                ("ln", nn.LayerNorm(config.mlp_out_size))
            ]))

        self.partial_loss_weights = config.loss_weights
        if config.pretrain_mode == 'RNA_ATAC':
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).float()


    def forward(
            self,
            rna_gene_ids: Optional[torch.Tensor] = None,
            rna_gene_values: Optional[torch.Tensor] = None,
            rna_id_labels: Optional[torch.Tensor] = None,
            rna_value_labels: Optional[torch.Tensor] = None,
            rna_mask: Optional[torch.Tensor] = None,
            rna_lengths: Optional[torch.Tensor] = None,
            rna_clusters: Optional[torch.Tensor] = None,
            rna_attention_mask: Optional[torch.Tensor] = None,
            sel_gene_ids: Optional[torch.Tensor] = None,
            sel_gene_states: Optional[torch.Tensor] = None,
            atac_attention_mask: Optional[torch.Tensor] = None,
            species: Optional[torch.Tensor] = None,
            atac_cell_peaks: Optional[torch.Tensor] = None,
            atac_peak_ids: Optional[torch.Tensor] = None,
            atac_peak_idfs: Optional[torch.Tensor] = None,
            atac_peak_idfs_masked: Optional[torch.Tensor] = None,
            atac_peak_acc_labels: Optional[torch.Tensor] = None,
            atac_mask: Optional[torch.Tensor] = None,
            sel_peak_ids: Optional[torch.Tensor] = None,
            sel_peak_idfs: Optional[torch.Tensor] = None,
            sel_peak_states: Optional[torch.Tensor] = None,
            atac_lengths: Optional[torch.Tensor] = None,
            atac_clusters: Optional[torch.Tensor] = None,
            peak_num: Optional[torch.Tensor] = None,
            modality: Optional[torch.Tensor] = None,
            cell_types: Optional[torch.Tensor] = None,
            cell_name: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor]]:
        pass


class TotalModel_downstream(TotalLossModel):
    def __init__(self, config, priors=None):
        config.mode = 'eval'
        super(TotalModel_downstream, self).__init__(config, priors=priors)
        # Initialize weights and apply final processing
        self.post_init()

    def match_forward(
            self,
            rna_gene_ids: Optional[torch.Tensor] = None,
            rna_gene_values: Optional[torch.Tensor] = None,
            rna_value_labels: Optional[torch.Tensor] = None,
            rna_mask: Optional[torch.Tensor] = None,
            rna_lengths: Optional[torch.Tensor] = None,
            rna_clusters: Optional[torch.Tensor] = None,
            rna_attention_mask: Optional[torch.Tensor] = None,
            atac_attention_mask: Optional[torch.Tensor] = None,
            species: Optional[torch.Tensor] = None,
            modality: Optional[torch.Tensor] = None,
            atac_cell_peaks: Optional[torch.Tensor] = None,
            atac_peak_ids: Optional[torch.Tensor] = None,
            atac_peak_idfs: Optional[torch.Tensor] = None,
            atac_peak_acc_labels: Optional[torch.Tensor] = None,
            atac_mask: Optional[torch.Tensor] = None,
            sel_peak_ids: Optional[torch.Tensor] = None,
            sel_peak_states: Optional[torch.Tensor] = None,
            atac_lengths: Optional[torch.Tensor] = None,
            atac_clusters: Optional[torch.Tensor] = None,
            cell_types: Optional[torch.Tensor] = None,
            cell_name: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            direct_out=False,
    ) -> Union[Tuple[torch.Tensor]]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        sequence_output_rna = self.encoder_rna.infer_forward(
            input_ids=rna_gene_ids,
            values=rna_gene_values,
            rna_mask=rna_mask,
            attention_mask=rna_attention_mask,
            species=species,
            modality=modality,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output_atac = self.encoder_atac.infer_forward(
            input_ids=atac_peak_ids,
            values=atac_peak_idfs,
            atac_mask=atac_mask,
            attention_mask=atac_attention_mask,
            species=species,
            modality=modality,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if direct_out:
            rna_cell_embs = sequence_output_rna[:, 0]
            atac_cell_embs = sequence_output_atac[:, 0]
        else:
            rna_cell_embs = self.rna_cell_projection(sequence_output_rna[:, 0])
            atac_cell_embs = self.atac_cell_projection(sequence_output_atac[:, 0])

        rna_embeddings = rna_cell_embs / rna_cell_embs.norm(dim=1, keepdim=True)
        atac_embeddings = atac_cell_embs / atac_cell_embs.norm(dim=1, keepdim=True)

        rna_res_dict, atac_res_dict = {}, {}
        for idx in range(rna_embeddings.shape[0]):
            rna_res_dict[cell_name[idx]] = rna_embeddings[idx].cpu()
            atac_res_dict[cell_name[idx]] = atac_embeddings[idx].cpu()

        similarity = (100.0 * rna_embeddings @ atac_embeddings.t()).softmax(dim=-1)
        values, indices = similarity.topk(1, dim=-1)

        return rna_res_dict, atac_res_dict, indices.squeeze().cpu().numpy().tolist(), np.arange(
            rna_embeddings.shape[0]).tolist()

    def get_rna_embeddings(
            self,
            rna_gene_ids: Optional[torch.Tensor] = None,
            rna_gene_values: Optional[torch.Tensor] = None,
            rna_value_labels: Optional[torch.Tensor] = None,
            rna_mask: Optional[torch.Tensor] = None,
            rna_lengths: Optional[torch.Tensor] = None,
            rna_clusters: Optional[torch.Tensor] = None,
            rna_attention_mask: Optional[torch.Tensor] = None,
            species: Optional[torch.Tensor] = None,
            modality: Optional[torch.Tensor] = None,
            cell_types: Optional[torch.Tensor] = None,
            cell_name: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            direct_out=False,
    ) -> Union[Tuple[torch.Tensor]]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        sequence_output_rna = self.encoder_rna.infer_forward(
            input_ids=rna_gene_ids,
            values=rna_gene_values,
            rna_mask=rna_mask,
            attention_mask=rna_attention_mask,
            species=species,
            modality=modality,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if direct_out:
            rna_cell_embs = sequence_output_rna[:, 0]
        else:
            rna_cell_embs = self.rna_cell_projection(sequence_output_rna[:, 0])

        rna_embeddings = rna_cell_embs / rna_cell_embs.norm(dim=1, keepdim=True)

        rna_res_dict, atac_res_dict = {}, {}
        for idx in range(rna_embeddings.shape[0]):
            rna_res_dict[cell_name[idx]] = rna_embeddings[idx].cpu()

        return rna_res_dict

    def get_atac_embeddings(
            self,
            atac_attention_mask: Optional[torch.Tensor] = None,
            species: Optional[torch.Tensor] = None,
            modality: Optional[torch.Tensor] = None,
            atac_cell_peaks: Optional[torch.Tensor] = None,
            atac_peak_ids: Optional[torch.Tensor] = None,
            atac_peak_idfs: Optional[torch.Tensor] = None,
            atac_peak_acc_labels: Optional[torch.Tensor] = None,
            atac_mask: Optional[torch.Tensor] = None,
            sel_peak_ids: Optional[torch.Tensor] = None,
            sel_peak_states: Optional[torch.Tensor] = None,
            atac_lengths: Optional[torch.Tensor] = None,
            atac_clusters: Optional[torch.Tensor] = None,
            cell_types: Optional[torch.Tensor] = None,
            cell_name: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            direct_out=False,
    ) -> Union[Tuple[torch.Tensor]]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        sequence_output_atac = self.encoder_atac.infer_forward(
            input_ids=atac_peak_ids,
            values=atac_peak_idfs,
            atac_mask=atac_mask,
            attention_mask=atac_attention_mask,
            species=species,
            modality=modality,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if direct_out:
            atac_cell_embs = sequence_output_atac[:, 0]
        else:
            atac_cell_embs = self.atac_cell_projection(sequence_output_atac[:, 0])
        atac_embeddings = atac_cell_embs / atac_cell_embs.norm(dim=1, keepdim=True)

        rna_res_dict, atac_res_dict = {}, {}
        for idx in range(atac_embeddings.shape[0]):
            atac_res_dict[cell_name[idx]] = atac_embeddings[idx].cpu()

        return atac_res_dict


