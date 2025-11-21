#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM, \
                         CLIPVisionModel, CLIPImageProcessor

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from src.model.gnn_models import GNN_Encoder

from torch_geometric.data import Data
import json
import os.path as osp
import glob

# DEFAULT_GRAPH_TOKEN = "<graph>"
# DEFAULT_G_START_TOKEN = "<g_start>"
# DEFAULT_G_END_TOKEN = "<g_end>"
DEFAULT_GRAPH_PATCH_TOKEN = 128002 # "<|reserved_special_token_0|>"
PADDING_TOKEN = 128001

num_to_token = {
0 : 15 ,
1 : 16 ,
2 : 17 ,
3 : 18 ,
4 : 19 ,
5 : 20 ,
6 : 21 ,
7 : 22 ,
8 : 23 ,
9 : 24 ,
10 : 605 ,
11 : 806 ,
12 : 717 ,
13 : 1032 ,
14 : 975 ,
15 : 868 ,
16 : 845 ,
17 : 1114 ,
18 : 972 ,
19 : 777 ,
20 : 508 ,
21 : 1691 ,
22 : 1313 ,
23 : 1419 ,
24 : 1187 ,
25 : 914 ,
26 : 1627 ,
27 : 1544 ,
28 : 1591 ,
29 : 1682 ,
30 : 966 ,
31 : 2148 ,
32 : 843 ,
33 : 1644 ,
34 : 1958 ,
35 : 1758 ,
36 : 1927 ,
37 : 1806 ,
38 : 1987 ,
39 : 2137 ,
40 : 1272 ,
41 : 3174 ,
42 : 2983 ,
43 : 3391 ,
44 : 2096 ,
45 : 1774 ,
46 : 2790 ,
47 : 2618 ,
48 : 2166 ,
49 : 2491 ,
}


class GraphLlamaConfig(LlamaConfig):
    model_type = "GraphLlama"

class GraphPretrainConfig:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)

def load_model_pretrained(model_name, pretrain_model_path): 
    # load conig json
    
    assert osp.exists(osp.join(pretrain_model_path, 'config.json')), 'config.json missing'
    with open(osp.join(pretrain_model_path, 'config.json'), 'r') as f:
        config_dict = json.load(f)
    args = GraphPretrainConfig(config_dict)
    model = model_name(args)
    pkl_files = glob.glob(osp.join(pretrain_model_path, '*.pkl'))
    state_dict = torch.load(pkl_files[0])
    # print(state_dict.keys())
    if 'logit_scale' in state_dict.keys(): 
        state_dict.pop('logit_scale')
    print('loading graph pre train model')
    model.load_state_dict(state_dict)


    return model, args

def transfer_param_tograph(clip_graph, gnn):
    
    print(clip_graph)
    gnn_state_dict = clip_graph.gnn.state_dict()
    gnn.load_state_dict(gnn_state_dict)
    return gnn

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=1, dropout=0.1, add_output_projection=False):
        super(CrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "Embedding dimension must be divisible by number of heads"
        
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.add_output_projection = add_output_projection
        if add_output_projection:
            self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query_seq, key_seq, value_seq, mask=None):
        batch_size, query_len, _ = query_seq.shape
        _, key_len, _ = key_seq.shape
        _, value_len, _ = value_seq.shape
        assert value_len == key_len, "Key and value sequences must have the same length"
        
        # Linear projections
        queries = self.query_proj(query_seq).view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        keys = self.key_proj(key_seq).view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.value_proj(value_seq).view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, values)
        output = output.transpose(1, 2).contiguous().view(batch_size, query_len, self.embed_dim)
        
        if self.add_output_projection:
            output = self.out_proj(output)
        return output

class GraphLlamaModel(LlamaModel):
    config_class = GraphLlamaConfig

    def __init__(self, config: LlamaConfig):
        super(GraphLlamaModel, self).__init__(config)

        # Only add graph tower using the initialize function
        # if hasattr(config, "graph_tower"):
        #     # HACK: for FSDP
        #     # self.vision_tower = [CLIPVisionModel.from_pretrained(config.graph_tower)]
        #     # self.arxiv_projector = nn.Linear(config.graph_hidden_size, config.hidden_size)
        #     if config.graph_tower == 'MPNN': 
        #         self.graph_tower = MPNN(in_channels = config.graph_hidden_size, hidden_channels = config.graph_hidden_size * 2, out_channels = config.graph_hidden_size, dropout = 0.1, num_layers = 2, if_param = False)
        # if hasattr(config, "use_graph_proj"):
        #     self.graph_projector = nn.Linear(config.graph_hidden_size, config.hidden_size)

    def get_graph_tower(self):
        graph_tower = getattr(self, 'graph_tower', None)
        if type(graph_tower) is list:
            graph_tower = graph_tower[0]
        return graph_tower

    def initialize_graph_modules(self, graph_tower, specs, cfg, use_cross_attn=True, 
                                 add_output_projection=True, 
                                 alignment_loss_weight=0.1,
                                 test_classifier_before_cross_attn=True, # only used for evaluating feature accuracy
                                 graph_select_layer=-1, pretrain_graph_mlp_adapter=None, fsdp=None):
        self.config.graph_tower = graph_tower

        if not hasattr(self, 'graph_tower'):
            graph_tower = GNN_Encoder(specs, cfg)
        else:
            graph_tower = self.graph_tower
        graph_tower.requires_grad_(True)

        if fsdp is not None and len(fsdp) > 0:
            self.graph_tower = [graph_tower]
        else:
            self.graph_tower = graph_tower

        self.config.use_graph_proj = True
        self.config.graph_select_layer = graph_select_layer

        self.graph_projector = nn.Linear(cfg.MODEL.HIDDEN_DIM, self.config.hidden_size)
        self.graph_classifier = nn.Linear(self.config.hidden_size, cfg.MODEL.NUM_CLASSES) # only used for evaluating feature accuracy
        self.alignment_loss_weight = alignment_loss_weight
        self.test_classifier_before_cross_attn = test_classifier_before_cross_attn

        if use_cross_attn:
            # This is setting up pseudo token embeddings for cross attention with graph features
            # self.graph_token_embeddings = nn.Embedding(num_soft_prompts, self.config.hidden_size)
            # for i in range(num_soft_prompts):
            #     self.graph_token_embeddings.weight.data[i] = self.embed_tokens.weight.data[num_to_token[i]].clone()
            self.graph_token_cross_attn = CrossAttention(self.config.hidden_size, num_heads=1, dropout=0.1, add_output_projection=add_output_projection)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        graph_data: Optional[Data] = None,
        original_input_ids: Optional[torch.LongTensor] = None,
        only_compute_graph_loss: Optional[bool] = False,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        # HACK: replace back original embeddings for LLaVA pretraining
        # orig_embeds_params = getattr(self, 'orig_embeds_params', None)
        # if orig_embeds_params is not None:
        #     orig_embeds_params = orig_embeds_params[0]
        #     with torch.no_grad():
        #         self.get_input_embeddings().weight.data[:-2] = orig_embeds_params[:-2].data

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            if original_input_ids is not None:
                original_input_embeds = self.embed_tokens(original_input_ids)

        graph_tower = self.get_graph_tower()
        if (graph_tower is not None) and (graph_data is not None): # and (input_ids.shape[1] != 1 or self.training)
            # encode graph features 
            graph_batch_ptr = graph_data.ptr
            graph_node_features = graph_tower(graph_data)
            dummy_graph_features = torch.zeros_like(graph_node_features, device=inputs_embeds.device)
            # project the graph node features to the hidden size of the model
            graph_node_features = self.graph_projector(graph_node_features)
            dummy_graph_features = self.graph_projector(dummy_graph_features)

            # compute graph classification loss
            graph_labels = graph_data.y
            graph_node_logits = self.graph_classifier(graph_node_features)
            graph_loss = F.cross_entropy(graph_node_logits, graph_labels)
            graph_accuracy = (graph_node_logits.detach().argmax(dim=1) == graph_labels).float().mean().cpu().item()
            
            new_input_embeds = []
            cur_graph_idx = 0
            graph_patch_token = DEFAULT_GRAPH_PATCH_TOKEN
            for cur_input_ids, cur_input_embeds in zip(input_ids, inputs_embeds):
                if (cur_input_ids == graph_patch_token).sum() == 0: # no patch token in the sequence
                    # this is training the bias of the projector
                    cur_input_embeds = cur_input_embeds + (0. * dummy_graph_features).sum()
                    new_input_embeds.append(cur_input_embeds)
                    cur_graph_idx += 1
                else:
                    cur_graph_features = graph_node_features[graph_batch_ptr[cur_graph_idx]:graph_batch_ptr[cur_graph_idx+1]]
                    num_patches = cur_graph_features.shape[0]

                    graph_patch_tokens = torch.where(cur_input_ids == graph_patch_token)[0]
                    assert len(graph_patch_tokens) == num_patches # make sure the number of graph patch tokens is the same as the number of patches

                    if hasattr(self, 'graph_token_cross_attn'):
                        # attend graph features with token embeddings
                        cur_text_embeddings = torch.cat(
                                (cur_input_embeds[:graph_patch_tokens[0]], 
                                cur_input_embeds[graph_patch_tokens[-1] + 1:]), dim=0)
                        cur_graph_features = torch.cat(
                                (cur_input_embeds[:graph_patch_tokens[0]], 
                                cur_graph_features,
                                cur_input_embeds[graph_patch_tokens[-1] + 1:]), dim=0)
                        cur_new_input_embeds = self.graph_token_cross_attn(cur_text_embeddings.unsqueeze(0).expand(1, -1, -1),
                                                                            cur_graph_features.unsqueeze(0).expand(1, -1, -1),
                                                                            cur_graph_features.unsqueeze(0).expand(1, -1, -1))
                        cur_new_input_embeds = cur_new_input_embeds.view(-1, self.config.hidden_size)
                        # fill in padding tokens
                        cur_new_input_embeds = torch.cat(
                                (self.embed_tokens(torch.tensor([PADDING_TOKEN]*num_patches, device=inputs_embeds.device)),
                                cur_new_input_embeds), dim=0)
                    else:
                        # insert the graph features after the graph start token
                        cur_new_input_embeds = torch.cat(
                            (cur_input_embeds[:graph_patch_tokens[0]], 
                                cur_graph_features, 
                                cur_input_embeds[graph_patch_tokens[-1] + 1:]), dim=0)
                    # if cur_graph_idx == 0:
                    #     print(cur_new_input_embeds[:32].abs().sum(dim=1))
                    cur_graph_idx += 1
                    new_input_embeds.append(cur_new_input_embeds)

            # compute distance to original embeddings
            graph_feature_distance = 0
            if original_input_ids is not None:
                pooled_graph_features = graph_node_features.view(cur_graph_idx, -1, graph_node_features.shape[-1]).mean(dim=1) # assuming the graph size is the same
                pooled_original_input_embeds = original_input_embeds.mean(dim=1)
                graph_feature_distance = F.mse_loss(pooled_graph_features, pooled_original_input_embeds)

            assert cur_graph_idx == len(graph_data.ptr)-1
            inputs_embeds = torch.stack(new_input_embeds, dim=0)

        if only_compute_graph_loss:
            outputs = CausalLMOutputWithPast(loss=graph_loss)
            outputs.graph_loss = graph_loss
            outputs.graph_accuracy = graph_accuracy
        else:
            outputs =  super(GraphLlamaModel, self).forward(
                input_ids=None, attention_mask=attention_mask, past_key_values=past_key_values, position_ids=position_ids,
                inputs_embeds=inputs_embeds, use_cache=use_cache, cache_position=cache_position,
                output_attentions=output_attentions, output_hidden_states=output_hidden_states, 
                return_dict=return_dict
            )
            outputs.alignment_loss = graph_feature_distance*self.alignment_loss_weight
        return outputs


class GraphLlamaForCausalLM(LlamaForCausalLM):
    config_class = GraphLlamaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = GraphLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def get_graph_tower(self):
        return self.get_model().get_graph_tower()

    def get_vision_tower(self):
        model = self.get_model()
        graph_tower = model.graph_tower
        if type(graph_tower) is list:
            graph_tower = graph_tower[0]
        return graph_tower
    
    def set_if_only_train_graph(self, only_train_graph):
        self.only_train_graph = only_train_graph

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        graph_data: Optional[Data] = None,
        original_input_ids: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            graph_data = graph_data,
            original_input_ids = original_input_ids,
            only_compute_graph_loss=self.only_train_graph,
            position_ids=position_ids,
            cache_position=cache_position,
        )

        if self.only_train_graph:
            return outputs

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        if original_input_ids is not None:
            loss += outputs.alignment_loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        **kwargs,
    ):
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids.contiguous()}  # `contiguous()` needed for compilation use cases

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "graph_data": kwargs.get("graph_data", None),
            }
        )
        return model_inputs

AutoConfig.register("GraphLlama", GraphLlamaConfig)
AutoModelForCausalLM.register(GraphLlamaConfig, GraphLlamaForCausalLM)
