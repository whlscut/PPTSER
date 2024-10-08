# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch LayoutLMv3 model. """
import math
from turtle import pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers import apply_chunking_to_forward, requires_backends
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    MaskedLMOutput,
    TokenClassifierOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.models.roberta.modeling_roberta import (
    RobertaIntermediate,
    RobertaLMHead,
    RobertaOutput,
    RobertaSelfOutput,
)
from transformers.utils import logging

from .configuration_layoutlmv3 import LayoutLMv3Config
from timm.models.layers import to_2tuple

logger = logging.get_logger(__name__)


class PromptEncoder(torch.nn.Module):
    def __init__(self, spell_length, hidden_size):
        super().__init__()
        self.spell_length = spell_length
        self.hidden_size = hidden_size
        self.n_layers = 2
        self.embedding = torch.nn.Embedding(self.spell_length, self.hidden_size)
        self.encode_layers = torch.nn.LSTM(input_size=self.hidden_size,
                                        hidden_size=self.hidden_size // 2,
                                        num_layers=self.n_layers,
                                        dropout=0.5,
                                        bidirectional=True,
                                        batch_first=True)
        
        self.mlp_head = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(self.hidden_size, self.hidden_size))
        logger.info("init prompt encoder...")

    def forward(self, seq_indices):

        input_embeds = self.embedding(seq_indices).unsqueeze(0)
        self.encode_layers.flatten_parameters()
        input_embeds = self.encode_layers(input_embeds)[0]
        
        output_embeds = self.mlp_head(input_embeds).squeeze()
        return output_embeds

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.num_patches_w = self.patch_shape[0]
        self.num_patches_h = self.patch_shape[1]

    def forward(self, x, position_embedding=None):
        x = self.proj(x)

        if position_embedding is not None:
            position_embedding = position_embedding.view(1, self.patch_shape[0], self.patch_shape[1], -1).permute(0, 3, 1, 2)
            Hp, Wp = x.shape[2], x.shape[3]
            position_embedding = F.interpolate(position_embedding, size=(Hp, Wp), mode='bicubic')
            x = x + position_embedding

        x = x.flatten(2).transpose(1, 2)
        return x

class LayoutLMv3Embeddings(nn.Module):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.  # okay
    """

    # Copied from transformers.models.bert.modeling_bert.BertEmbeddings.__init__
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        '''
        self.label_head_locate = config.label_head_locate
        self.total_label_len = config.total_len_of_all_tokenized_label
        if self.total_label_len:
            self.label_type_embeddings = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        '''

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

        self.x_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.coordinate_size)
        self.y_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.coordinate_size)
        self.h_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.shape_size)
        self.w_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.shape_size)

    def _calc_spatial_position_embeddings(self, bbox):
        try:
            assert torch.all(0 <= bbox) and torch.all(bbox <= 1023)
            left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
            upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
            right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
            lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])
        except IndexError as e:
            raise IndexError("The :obj:`bbox` coordinate values should be within 0-1000 range.") from e

        h_position_embeddings = self.h_position_embeddings(torch.clip(bbox[:, :, 3] - bbox[:, :, 1], 0, 1023))
        w_position_embeddings = self.w_position_embeddings(torch.clip(bbox[:, :, 2] - bbox[:, :, 0], 0, 1023))

        spatial_position_embeddings = torch.cat(
            [
                left_position_embeddings,
                upper_position_embeddings,
                right_position_embeddings,
                lower_position_embeddings,
                h_position_embeddings,
                w_position_embeddings,
            ],
            dim=-1,
        )
        return spatial_position_embeddings

    def create_position_ids_from_input_ids(self, input_ids, padding_idx, past_key_values_length=0):
        """
        Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
        are ignored. This is modified from fairseq's `utils.make_positions`.

        Args:
            x: torch.Tensor x:

        Returns: torch.Tensor
        """
        # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
        mask = input_ids.ne(padding_idx).int()  # ne: not equal
        incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
        return incremental_indices.long() + padding_idx

    def forward(
        self,
        input_ids=None,
        bbox=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        past_key_values_length=0,
    ):
        if position_ids is None:
            if input_ids is not None:
                position_ids = self.create_position_ids_from_input_ids(
                    input_ids, self.padding_idx, past_key_values_length).to(input_ids.device)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings

        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings

        spatial_position_embeddings = self._calc_spatial_position_embeddings(bbox)

        embeddings = embeddings + spatial_position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: torch.Tensor≈

        Returns: torch.Tensor
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)


class LayoutLMv3PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = LayoutLMv3Config
    base_model_prefix = "layoutlmv3"

    # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class LayoutLMv3SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.has_relative_attention_bias = config.has_relative_attention_bias
        self.has_spatial_attention_bias = config.has_spatial_attention_bias

        self.label_head_locate = config.label_head_locate
        self.customized_selfattn = False
    
    def set_customized_selfattn(self, status=False):
        self.customized_selfattn = status
        if self.customized_selfattn:
            del self.value

    def customized_dropout(self, inputs, dropout_rate=0):
        assert dropout_rate >= 0
        if not self.training or dropout_rate==0:
            return inputs
        else:
            mask = torch.bernoulli(inputs, p=dropout_rate).bool().to(inputs.device)
            inputs[mask] = 0
            return inputs
        

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def head_norm(self, inputs, epsilon=1e-7):
        mean = inputs.mean(dim=1, keepdim=True)
        std = inputs.std(dim=1, keepdim=True)
        outputs = (inputs-mean)# / (std + epsilon)
        return outputs

    def cogview_attn(self, attention_scores, alpha=32):
        '''
        https://arxiv.org/pdf/2105.13290.pdf
        Section 2.4 Stabilization of training: Precision Bottleneck Relaxation (PB-Relax).
        A replacement of the original nn.Softmax(dim=-1)(attention_scores)
        Seems the new attention_probs will result in a slower speed and a little bias
        Can use torch.allclose(standard_attention_probs, cogview_attention_probs, atol=1e-08) for comparison
        The smaller atol (e.g., 1e-08), the better.
        '''
        if self.customized_selfattn:
            scaled_attention_scores = attention_scores
            new_attention_scores = self.customized_dropout(scaled_attention_scores, dropout_rate=0)
            return new_attention_scores
        else:
            scaled_attention_scores = attention_scores / alpha
            max_value = scaled_attention_scores.amax(dim=(-1)).unsqueeze(-1)
            new_attention_scores = (scaled_attention_scores - max_value) * alpha
            return nn.Softmax(dim=-1)(new_attention_scores)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        rel_pos=None,
        rel_2d_pos=None,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states)) if not self.customized_selfattn \
                else None
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states)) if not self.customized_selfattn \
                else None
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2) if not self.customized_selfattn \
                else None
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states)) if not self.customized_selfattn \
                else None

        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # The attention scores QT K/√d could be significantly larger than input elements, and result in overflow.
        # Changing the computational order into QT(K/√d) alleviates the problem. (https://arxiv.org/pdf/2105.13290.pdf)
        if not self.customized_selfattn:
            attention_scores = torch.matmul(query_layer / math.sqrt(self.attention_head_size), key_layer.transpose(-1, -2))
        else:
            query_layer = F.normalize(query_layer, dim=-1)
            key_layer = F.normalize(key_layer, dim=-1)
            query_layer = self.dropout(query_layer); key_layer = self.dropout(key_layer)
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) / 0.15

            attention_probs = attention_scores
            label_head_locate = torch.tensor(self.label_head_locate).to(attention_probs.device)
            output = attention_probs[:, :, label_head_locate, :]
            output = output.amax(dim=1)

            return output

        if self.has_relative_attention_bias and self.has_spatial_attention_bias:
            attention_scores += (rel_pos + rel_2d_pos) / math.sqrt(self.attention_head_size)
        elif self.has_relative_attention_bias:
            attention_scores += rel_pos / math.sqrt(self.attention_head_size)

        # attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
            attention_scores = attention_scores + attention_mask

        attention_probs = self.cogview_attn(attention_scores)
 
        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        
        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class LayoutLMv3Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = LayoutLMv3SelfAttention(config)
        self.output = RobertaSelfOutput(config)
        self.pruned_heads = set()

        self.customized_attn = False
    
    def set_customized_attn(self, status=False):
        self.customized_attn = status
        if self.customized_attn:
            del self.output
        self.self.set_customized_selfattn(self.customized_attn)

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        rel_pos=None,
        rel_2d_pos=None,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
            rel_pos=rel_pos,
            rel_2d_pos=rel_2d_pos,
        )
        if self.customized_attn:
            return (self_outputs,)

        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]

        return outputs


class LayoutLMv3Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = LayoutLMv3Attention(config)
        assert not config.is_decoder and not config.add_cross_attention, \
            "This version do not support decoder. Please refer to RoBERTa for implementation of is_decoder."
        self.intermediate = RobertaIntermediate(config)
        self.output = RobertaOutput(config)

        self.customized_layer = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        rel_pos=None,
        rel_2d_pos=None,
    ):
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
            rel_pos=rel_pos,
            rel_2d_pos=rel_2d_pos,
        )
        if self.customized_layer:
            return self_attention_outputs
            
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[1:]

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs
        return outputs

    def set_customized_layer(self, status=False):
        self.customized_layer = status
        if self.customized_layer:
            del self.intermediate
            del self.output
        self.attention.set_customized_attn(self.customized_layer)

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class LayoutLMv3Encoder(nn.Module):
    def __init__(self, config, detection=False, out_features=None):
        super().__init__()
        self.config = config
        self.detection = detection
        self.layer = nn.ModuleList([LayoutLMv3Layer(config) for _ in range(config.num_hidden_layers)])
        self.customized_encoder = False
        self.gradient_checkpointing = False

        self.has_relative_attention_bias = config.has_relative_attention_bias
        self.has_spatial_attention_bias = config.has_spatial_attention_bias

        self.k = 0
        self.len_of_all_tokenized_label = config.len_of_all_tokenized_label
        self.total_label_len = sum(self.len_of_all_tokenized_label)
        self.max_len = 512
        if self.len_of_all_tokenized_label:
            self.label_relative_position = torch.cat([torch.arange(self.max_len - i, self.max_len) \
                                                      for i in self.len_of_all_tokenized_label])

        if self.has_relative_attention_bias:
            self.rel_pos_bins = config.rel_pos_bins
            self.max_rel_pos = config.max_rel_pos
            self.rel_pos_onehot_size = config.rel_pos_bins
            self.rel_pos_bias = nn.Linear(self.rel_pos_onehot_size, config.num_attention_heads, bias=False)

        if self.has_spatial_attention_bias:
            self.max_rel_2d_pos = config.max_rel_2d_pos
            self.rel_2d_pos_bins = config.rel_2d_pos_bins
            self.rel_2d_pos_onehot_size = config.rel_2d_pos_bins

            self.rel_pos_x_bias = nn.Linear(self.rel_2d_pos_onehot_size, config.num_attention_heads, bias=False)
            self.rel_pos_y_bias = nn.Linear(self.rel_2d_pos_onehot_size, config.num_attention_heads, bias=False)

        if self.detection:
            self.gradient_checkpointing = True
            embed_dim = self.config.hidden_size
            self.out_features = out_features
            self.out_indices = [int(name[5:]) for name in out_features]
            self.fpn1 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                # nn.SyncBatchNorm(embed_dim),
                nn.BatchNorm2d(embed_dim),
                nn.GELU(),
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            )

            self.fpn2 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            )

            self.fpn3 = nn.Identity()

            self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
    
    def set_customized_encoder(self, status):
        self.customized_encoder = status
        self.layer[-1].set_customized_layer(self.customized_encoder)

    def relative_position_bucket(self, relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        ret = 0
        if bidirectional:
            num_buckets //= 2
            ret += (relative_position > 0).long() * num_buckets
            n = torch.abs(relative_position)
        else:
            n = torch.max(-relative_position, torch.zeros_like(relative_position))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
                torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).to(torch.long)
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def _cal_1d_pos_emb(self, hidden_states, position_ids, valid_span,
                        mask=None):
        VISUAL_NUM = 196 + 1

        rel_pos_mat = position_ids.unsqueeze(-2) - position_ids.unsqueeze(-1)

        if valid_span is not None:
            rel_pos_mat[(rel_pos_mat > 0) & (valid_span == False)] = position_ids.shape[1]
            rel_pos_mat[(rel_pos_mat < 0) & (valid_span == False)] = -position_ids.shape[1]

            rel_pos_mat[:, -VISUAL_NUM:, :-VISUAL_NUM] = 0
            rel_pos_mat[:, :-VISUAL_NUM, -VISUAL_NUM:] = 0

        rel_pos = self.relative_position_bucket(
            rel_pos_mat,
            num_buckets=self.rel_pos_bins,
            max_distance=self.max_rel_pos,
        )

        rel_pos = F.one_hot(rel_pos, num_classes=self.rel_pos_onehot_size).type_as(hidden_states)
        rel_pos = self.rel_pos_bias(rel_pos).permute(0, 3, 1, 2)
        rel_pos = rel_pos.contiguous()
        return rel_pos

    def _cal_2d_pos_emb(self, hidden_states, bbox):
        position_coord_x = bbox[:, :, 0]
        position_coord_y = bbox[:, :, 3]
        rel_pos_x_2d_mat = position_coord_x.unsqueeze(-2) - position_coord_x.unsqueeze(-1)
        rel_pos_y_2d_mat = position_coord_y.unsqueeze(-2) - position_coord_y.unsqueeze(-1)
        rel_pos_x = self.relative_position_bucket(
            rel_pos_x_2d_mat,
            num_buckets=self.rel_2d_pos_bins,
            max_distance=self.max_rel_2d_pos,
        )
        rel_pos_y = self.relative_position_bucket(
            rel_pos_y_2d_mat,
            num_buckets=self.rel_2d_pos_bins,
            max_distance=self.max_rel_2d_pos,
        )

        rel_pos_x = F.one_hot(rel_pos_x, num_classes=self.rel_2d_pos_onehot_size).type_as(hidden_states)
        rel_pos_y = F.one_hot(rel_pos_y, num_classes=self.rel_2d_pos_onehot_size).type_as(hidden_states)
        rel_pos_x = self.rel_pos_x_bias(rel_pos_x).permute(0, 3, 1, 2)
        rel_pos_y = self.rel_pos_y_bias(rel_pos_y).permute(0, 3, 1, 2)
        rel_pos_x = rel_pos_x.contiguous()
        rel_pos_y = rel_pos_y.contiguous()
        rel_2d_pos = rel_pos_x + rel_pos_y
        return rel_2d_pos

    def forward(
        self,
        hidden_states,
        bbox=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True, 
        position_ids=None,
        Hp=None,
        Wp=None,
        valid_span=None,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None

        rel_pos = self._cal_1d_pos_emb(hidden_states, position_ids, valid_span, mask=attention_mask) if self.has_relative_attention_bias else None
        rel_2d_pos = self._cal_2d_pos_emb(hidden_states, bbox) if self.has_spatial_attention_bias else None

        if self.detection:
            feat_out = {}
            j = 0

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                    rel_pos,
                    rel_2d_pos
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                    rel_pos=rel_pos,
                    rel_2d_pos=rel_2d_pos,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

            if self.detection and i in self.out_indices:
                xp = hidden_states[:, -Hp*Wp:, :].permute(0, 2, 1).reshape(len(hidden_states), -1, Hp, Wp)
                feat_out[self.out_features[j]] = self.ops[j](xp.contiguous())
                j += 1

        if self.detection:
            return feat_out

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class LayoutLMv3Model(LayoutLMv3PreTrainedModel):
    """
    well the label
    """
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    # Copied from transformers.models.bert.modeling_bert.BertModel.__init__ with Bert->Roberta
    def __init__(self, config, detection=False, out_features=None, image_only=False):
        super().__init__(config)
        self.config = config
        assert not config.is_decoder and not config.add_cross_attention, \
            "This version do not support decoder. Please refer to RoBERTa for implementation of is_decoder."
        self.detection = detection
        if not self.detection:
            self.image_only = False
        else:
            assert config.visual_embed
            self.image_only = image_only

        if not self.image_only:
            self.embeddings = LayoutLMv3Embeddings(config)
        self.encoder = LayoutLMv3Encoder(config, detection=detection, out_features=out_features)

        if config.visual_embed:
            embed_dim = self.config.hidden_size
            self.patch_embed = PatchEmbed(embed_dim=embed_dim)

            patch_size = 16
            size = int(self.config.input_size / patch_size)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pos_embed = nn.Parameter(torch.zeros(1, size * size + 1, embed_dim))  
            self.pos_drop = nn.Dropout(p=0.)

            self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.dropout = nn.Dropout(config.hidden_dropout_prob)

            if self.config.has_relative_attention_bias or self.config.has_spatial_attention_bias:
                self._init_visual_bbox(img_size=(size, size))

            from functools import partial
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
            self.norm = norm_layer(embed_dim)
        
        self.max_len = 512
        self.len_of_all_tokenized_label = config.len_of_all_tokenized_label
        self.label_head_locate = config.label_head_locate

        self.init_weights()

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

    def _init_visual_bbox(self, img_size=(14, 14), max_len=1000):
        visual_bbox_x = torch.div(torch.arange(0, max_len * (img_size[1] + 1), max_len),
                                  img_size[1], rounding_mode='trunc')
        visual_bbox_y = torch.div(torch.arange(0, max_len * (img_size[0] + 1), max_len),
                                  img_size[0], rounding_mode='trunc')
        visual_bbox = torch.stack(
            [
                visual_bbox_x[:-1].repeat(img_size[0], 1),
                visual_bbox_y[:-1].repeat(img_size[1], 1).transpose(0, 1),
                visual_bbox_x[1:].repeat(img_size[0], 1),
                visual_bbox_y[1:].repeat(img_size[1], 1).transpose(0, 1),
            ],
            dim=-1,
        ).view(-1, 4)

        cls_token_box = torch.tensor([[0 + 1, 0 + 1, max_len - 1, max_len - 1]])
        self.visual_bbox = torch.cat([cls_token_box, visual_bbox], dim=0)

    def _calc_visual_bbox(self, device, dtype, bsz):
        visual_bbox = self.visual_bbox.repeat(bsz, 1, 1)
        visual_bbox = visual_bbox.to(device).type(dtype)
        return visual_bbox

    def forward_image(self, x):
        if self.detection:
            x = self.patch_embed(x, self.pos_embed[:, 1:, :] if self.pos_embed is not None else None)
        else:
            x = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        if self.pos_embed is not None and self.detection:
            cls_tokens = cls_tokens + self.pos_embed[:, :1, :]

        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None and not self.detection:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        x = self.norm(x)
        return x

    def get_extended_attention_mask(self, attention_mask, input_shape, device, label_len=None,):
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                # in case past_key_values are used we need to add a prefix ones mask to the causal mask
                # causal and attention masks must have same type with pytorch version < 1.3
                causal_mask = causal_mask.to(attention_mask.dtype)

                if causal_mask.shape[1] < attention_mask.shape[1]:
                    prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
                    causal_mask = torch.cat(
                        [
                            torch.ones(
                                (batch_size, seq_length, prefix_seq_len), device=device, dtype=causal_mask.dtype
                            ),
                            causal_mask,
                        ],
                        axis=-1,
                    )

                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                row_mask = attention_mask[:, None, :, None]
                col_mask = attention_mask[:, None, None, :]
                extended_attention_mask = row_mask & col_mask
                
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask
        

    # Copied from transformers.models.bert.modeling_bert.BertModel.forward
    def forward(
        self,
        input_ids=None,
        bbox=None,
        attention_mask=None,
        token_type_ids=None,
        valid_span=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        images=None,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        use_cache = False

        # if input_ids is not None and inputs_embeds is not None:
        #     raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
            device = input_ids.device
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
            device = inputs_embeds.device
        elif images is not None:
            batch_size = len(images)
            device = images.device
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds or images")

        if not self.image_only:
            # past_key_values_length
            past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

            if attention_mask is None:
                attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
            if token_type_ids is None:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        # extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        if not self.image_only:
            if bbox is None:
                bbox = torch.zeros(tuple(list(input_shape) + [4]), dtype=torch.long, device=device)
            embedding_output = self.embeddings(
                input_ids=input_ids,
                bbox=bbox,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
                inputs_embeds=inputs_embeds,
                past_key_values_length=past_key_values_length,
            )

        final_bbox = final_position_ids = None
        Hp = Wp = None
        if images is not None:
            patch_size = 16
            Hp, Wp = int(images.shape[2] / patch_size), int(images.shape[3] / patch_size)
            visual_emb = self.forward_image(images)
            if self.detection:
                visual_attention_mask = torch.ones((batch_size, visual_emb.shape[1]), dtype=torch.long, device=device)
                if self.image_only:
                    attention_mask = visual_attention_mask
                else:
                    attention_mask = torch.cat([attention_mask, visual_attention_mask], dim=1)
            elif self.image_only:
                attention_mask = torch.ones((batch_size, visual_emb.shape[1]), dtype=torch.long, device=device)

            if self.config.has_relative_attention_bias or self.config.has_spatial_attention_bias:
                if self.config.has_spatial_attention_bias:
                    visual_bbox = self._calc_visual_bbox(device, dtype=torch.long, bsz=batch_size)
                    if self.image_only:
                        final_bbox = visual_bbox
                    else:
                        final_bbox = torch.cat([bbox, visual_bbox], dim=1)

                visual_position_ids = torch.arange(0, visual_emb.shape[1], dtype=torch.long, device=device).repeat(
                    batch_size, 1)
                if self.image_only:
                    final_position_ids = visual_position_ids
                else:
                    # if position_ids is None:
                    position_ids = torch.arange(0, input_shape[1], device=device).unsqueeze(0)
                    position_ids = position_ids.expand_as(input_ids)
                    final_position_ids = torch.cat([position_ids, visual_position_ids], dim=1)

            if self.image_only:
                embedding_output = visual_emb
            else:
                embedding_output = torch.cat([embedding_output, visual_emb], dim=1)
            embedding_output = self.LayerNorm(embedding_output)
            embedding_output = self.dropout(embedding_output)
        elif self.config.has_relative_attention_bias or self.config.has_spatial_attention_bias:
            if self.config.has_spatial_attention_bias:
                final_bbox = bbox
            if self.config.has_relative_attention_bias:
                position_ids = self.embeddings.position_ids[:, :input_shape[1]]
                position_ids = position_ids.expand_as(input_ids)
                final_position_ids = position_ids
        extended_attention_mask: torch.Tensor = \
            self.get_extended_attention_mask(attention_mask, None, device,
                                             self.len_of_all_tokenized_label,)

        encoder_outputs = self.encoder(
            embedding_output,
            bbox=final_bbox,
            position_ids=final_position_ids,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            Hp=Hp,
            Wp=Wp,
            valid_span=valid_span,
        )

        if self.detection:
            return encoder_outputs

        sequence_output = encoder_outputs[0]
        pooled_output = None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]
        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

class LayoutLMv3ForFSMethod(LayoutLMv3PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        torch.autograd.set_detect_anomaly(True)
        self.num_labels = config.num_labels

        self.layoutlmv3 = LayoutLMv3Model(config)

        self.init_weights()

        self.layoutlmv3.encoder.set_customized_encoder(True)

        self.k = 0

    def forward(
        self,
        input_ids=None,
        bbox=None,
        attention_mask=None, 
        token_type_ids=None, 
        position_ids=None, 
        valid_span=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        images=None,
        segment_ids=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.layoutlmv3(
            input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            images=images,
            valid_span=valid_span,
        )

        sequence_output = outputs[0]

        logits = sequence_output.transpose(1, 2)
        logits_fct = [logits[..., :-2], logits[..., -2:]]
        labels_fct = [labels[:, 0, :], labels[:, 1, :]]

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = 0

            loss_ratio = [1, .1]
            for i, (logit, label) in enumerate(zip(logits_fct, labels_fct)):
                # Only keep active parts of the loss
                label_mask = label != -100
                if torch.all(~label_mask):
                    continue
                logit_masked = logit[label_mask]
                label_masked = label[label_mask]
                # overall loss
                loss += loss_ratio[i] * loss_fct(logit_masked, label_masked)

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        else:
            return TokenClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )