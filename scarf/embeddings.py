from typing import Optional
import torch
import torch.nn as nn
from collections import OrderedDict


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, length, kernel_size, stride_size):
        super(ConvBlock, self).__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride_size, (kernel_size - stride_size) // 2)
        self.layer_norm = nn.LayerNorm([out_channels, length])
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1d(x)
        x = self.layer_norm(x)
        x = self.relu(x)
        return x


class Peak_Network(nn.Module):
    def __init__(self, in_channels, length, channel_list):
        super(Peak_Network, self).__init__()
        # list1: in32-64-128-256-512 peak_dim=32
        # list2: in3-16-64-256-256
        self.block1 = ConvBlock(in_channels, channel_list[0], length//4, 16, 4)
        self.block2 = ConvBlock(channel_list[0], channel_list[1], length//16, 16, 4)
        self.block3 = ConvBlock(channel_list[1], channel_list[2], length//64, 16, 4)
        self.block4 = ConvBlock(channel_list[2], channel_list[3], length//128, 8, 2)
        self.avg_pool = nn.AvgPool1d(13)

    def forward(self, x):
        x = x.permute(0, 1, 3, 2)
        B, L, dim, peak_num = x.shape
        x = x.view(-1, dim, peak_num)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.avg_pool(x)
        return x.squeeze()



class ContinuousValueEncoder(nn.Module):
    """
    Encode real number values to a vector using neural nets projection.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_value: int = 255):
        super().__init__()
        self.max_value = max_value
        self.linear1 = nn.Linear(1, d_model)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len]
        """
        # TODO: test using actual embedding layer if input is categorical
        # expand last dimension
        x = x.unsqueeze(-1)
        x = x.float().cuda()
        # clip x to [-inf, max_value]
        x = torch.clamp(x, max=self.max_value)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.norm(x)
        x = self.dropout(x)
        return x


class BertEmbeddings(nn.Module):
    def __init__(self, config, priors, data_type='RNA', word_embeddings=None):
        super().__init__()
        self.hidden_size = config.rna_model_cfg['hidden_size'] if data_type=='RNA' else config.atac_model_cfg['hidden_size']
        if word_embeddings is None:
            if data_type == 'RNA':
                self.word_embeddings = nn.Embedding(config.rna_model_cfg['vocab_size'], self.hidden_size, padding_idx=config.pad_token_id)
            else:
                self.word_embeddings = nn.Embedding(config.peak_total_num, self.hidden_size,
                                                    padding_idx=config.pad_token_id)
        self.data_type = data_type
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_size))

        # value编码
        self.value_dim = 1
        self.use_value_emb = config.to_dict().get('use_value_emb', False)
        if self.use_value_emb:
            self.values_embeddings = ContinuousValueEncoder(self.hidden_size)
            self.value_dim = self.hidden_size

        self.concat_dim = self.hidden_size + self.value_dim
        self.concat_embeddings = nn.Sequential(OrderedDict([
            ("cat_fc", nn.Linear(self.concat_dim, self.hidden_size)),
            ("cat_ln", nn.LayerNorm(self.hidden_size)),
            ("cat_gelu", QuickGELU()),
            ("cat_proj", nn.Linear(self.hidden_size, self.hidden_size))
        ]))
        self.mode = config.to_dict().get('mode', 'train')

        # 以下是原始 BertEmbedding 部分（除去word_embeddings）
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, self.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, self.hidden_size)

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

        # species embs
        self.config = config
        self.species_embeddings = nn.Embedding(config.species_num, self.hidden_size)
        self.modality_embeddings = nn.Embedding(config.modality_number, self.hidden_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_size))

    def get_peak_embs(self, input):
        return self.word_embeddings(input)

    def get_gene_embs(self, input):
        return self.word_embeddings(input)

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            values: Optional[torch.FloatTensor] = None,
            species: Optional[torch.FloatTensor] = None,
            modality: Optional[torch.FloatTensor] = None,
            gene_peaks: Optional[torch.FloatTensor] = None,
            atac_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            past_key_values_length: int = 0,
    ) -> torch.Tensor:


        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length: seq_length + 3 + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        # value concat
        if self.use_value_emb or self.data_type == 'ATAC':
            values = self.values_embeddings(values)
        else:
            values = values.unsqueeze(-1).float()
        inputs_embeds = torch.cat([inputs_embeds, values], dim=2)
        inputs_embeds = self.concat_embeddings(inputs_embeds)

        # 以下是原始 BertEmbedding 部分
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings

        B, L = inputs_embeds.shape[0:2]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        species_tokens = self.species_embeddings(species).unsqueeze(1)
        modality_tokens = self.modality_embeddings(modality).unsqueeze(1)

        if self.config.use_middle_cls_token:
            token_position = L//2
            embeddings = torch.cat((embeddings[:, :token_position, :], cls_tokens, species_tokens, modality_tokens,embeddings[:, token_position:, :]), dim=1)
        else:
            embeddings = torch.cat((cls_tokens, species_tokens, modality_tokens, embeddings), dim=1)

        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings