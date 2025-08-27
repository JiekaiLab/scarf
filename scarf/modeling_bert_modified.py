from typing import List, Optional, Tuple, Union
import torch
from torch import nn
from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
)
from transformers.utils import logging
from transformers.models.bert.modeling_bert import (
    BertEncoder,
    BertPooler,
    BertPredictionHeadTransform,
    BertOnlyMLMHead,
    BertPreTrainedModel,
)
from .embeddings import BertEmbeddings
from transformers import BertConfig
from .mamba_modified import MambaEncoder


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

logger = logging.get_logger(__name__)

class BertModel(BertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config, priors, add_pooling_layer=True, data_type='RNA'):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config, priors, data_type=data_type)
        self.config.use_cls_token = config.to_dict().get('use_cls_token', False)

        self.model_config = config.rna_model_cfg if data_type == 'RNA' else config.atac_model_cfg
        if self.model_config['encoder_type'] == 'bert':
            self.model_config = BertConfig(**(self.model_config))
            self.encoder = BertEncoder(self.model_config)
        elif self.model_config['encoder_type'] == 'mamba':
            self.model_config = BertConfig(**(self.model_config))
            self.encoder = MambaEncoder(self.model_config)
        self.data_type = data_type
        self.pooler = BertPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            values: Optional[torch.Tensor] = None,
            gene_peaks: Optional[torch.Tensor] = None,
            atac_mask: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            species: Optional[torch.Tensor] = None,
            modality: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if self.config.use_cls_token:
            attention_mask = torch.cat([torch.ones(batch_size, 3).to(device), attention_mask], dim=1)
        else:
            attention_mask = torch.cat([torch.ones(batch_size, 2).to(device), attention_mask], dim=1)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            values=values,
            species=species,
            modality=modality,
            gene_peaks=gene_peaks,
            atac_mask=atac_mask,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        if self.model_config.encoder_type == 'bert':
            encoder_outputs = self.encoder(
                embedding_output,
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            sequence_output = encoder_outputs[0]
        elif self.model_config.encoder_type == 'mamba' and self.model_config.mamba_version == 'v_hf':
            encoder_outputs = self.encoder(
                inputs_embeds=embedding_output,
                attention_mask=extended_attention_mask,
                use_cache=use_cache,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            sequence_output = encoder_outputs[0]
        elif self.model_config.encoder_type == 'mamba' and self.model_config.mamba_version == 'v_manual':
            sequence_output = self.encoder(embedding_output)
        if self.config.use_middle_cls_token:
            B, L = sequence_output.shape[0:2]
            token_position = (L-3) // 2
            sequence_output = torch.cat((sequence_output[:, token_position:token_position+3, :],
                                         sequence_output[:, :token_position, :],
                                         sequence_output[:, token_position+3:, :]), dim=1)
        # print(sequence_output[:, 1:4, 1])
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None  # use_cls token
        # pooled_output = self.pooler(sequence_output[:,1:,:]) if self.pooler is not None else None # use_first token
        if not return_dict:
            return (sequence_output, pooled_output) #+ encoder_outputs[1:]

        if self.model_config.encoder_type == 'bert':
            return BaseModelOutputWithPoolingAndCrossAttentions(
                last_hidden_state=sequence_output,
                pooler_output=pooled_output,
                past_key_values=encoder_outputs.past_key_values,
                hidden_states=encoder_outputs.hidden_states,
                attentions=encoder_outputs.attentions,
                cross_attentions=encoder_outputs.cross_attentions,
            )
        else:
            return BaseModelOutputWithPoolingAndCrossAttentions(
                last_hidden_state=sequence_output,
                pooler_output=pooled_output,
                past_key_values=None,
                hidden_states=None,
                attentions=None,
                cross_attentions=None,
            )


class BertForMaskedLMWithRNA(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias", r"cls.predictions.decoder.weight"]

    def __init__(self, config, priors):
        config.max_input_size = config.rna_model_cfg['rna_max_input_size']
        config.max_position_embeddings = config.rna_model_cfg['rna_pos_max_size'] + 3
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        for _ in range(10):
            print('Now using our modified model!!!')

        self.bert = BertModel(config,
                              priors,
                              add_pooling_layer=False,
                              data_type='RNA')

        # TODO: if values are used, than outputs need to be ids and values ---> change to self.cls_and_reg = BertOnlyMLMHead
        self.cls = BertOnlyMLMHead(BertConfig(**(config.rna_model_cfg)))
        print(config.rna_model_cfg)

        self.use_values = config.use_values
        self.use_cls_token = config.to_dict().get('use_cls_token', False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def infer_forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            values: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            species: Optional[torch.Tensor] = None,
            modality: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            rna_mask: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids=input_ids,
            values=values,
            attention_mask=attention_mask,
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

        sequence_output = outputs[0].float()
        return sequence_output


class BertForMaskedLMWithATAC(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias", r"cls.predictions.decoder.weight"]

    def __init__(self, config, priors):
        config.max_input_size = config.atac_model_cfg['atac_max_input_size']
        config.max_position_embeddings = config.atac_model_cfg['atac_pos_max_size'] + 3
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        for _ in range(10):
            print('Now using ATAC model!!!')

        self.bert = BertModel(config,
                              priors,
                              add_pooling_layer=False,
                              data_type='ATAC')

        # TODO: if values are used, than outputs need to be ids and values ---> change to self.cls_and_reg = BertOnlyMLMHead
        self.cls = BertOnlyMLMHead(BertConfig(**(config.atac_model_cfg)))
        model_config = config.atac_model_cfg
        original_vocab_size = model_config['vocab_size']
        model_config.update({"vocab_size": 1})
        self.cls4peak_acc = BertOnlyMLMHead(BertConfig(**(model_config)))
        model_config.update({"vocab_size": original_vocab_size})

        self.model_config = model_config
        self.use_cls_token = config.to_dict().get('use_cls_token', False)
        self.config = config
        # Initialize weights and apply final processing
        self.post_init()

    def get_peak_embs(self, input):
        return self.bert.embeddings.get_peak_embs(input, data_type='ATAC')

    def infer_forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            values: Optional[torch.Tensor] = None,
            atac_mask: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            species: Optional[torch.Tensor] = None,
            modality: Optional[torch.Tensor] = None,
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
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids=input_ids,
            values=values,
            atac_mask=atac_mask,
            attention_mask=attention_mask,
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

        sequence_output = outputs[0]

        return sequence_output
