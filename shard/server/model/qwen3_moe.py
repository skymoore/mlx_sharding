from dataclasses import dataclass
from typing import Optional, Any

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.qwen3_moe import ModelArgs as BaseModelArgs, Qwen3MoeDecoderLayer
from .base import IdentityBlock


@dataclass
class ModelArgs(BaseModelArgs):
    start_layer: int = 0
    end_layer: int = 12


class LanguageModel(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        self.start_layer = config.start_layer
        self.end_layer = config.end_layer
        
        if self.start_layer == 0:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        self.layers = []
        for i in range(self.num_hidden_layers):
            if self.start_layer <= i < self.end_layer:
                self.layers.append(Qwen3MoeDecoderLayer(config, i))
            else:
                self.layers.append(IdentityBlock())
        
        if self.end_layer == self.num_hidden_layers:
            self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        if self.start_layer == 0:
            h = self.embed_tokens(x)
        else:
            h = x

        if cache is None:
            cache = [None] * len(self.layers)

        mask = None
        T = h.shape[1]
        if T > 1:
            from mlx_lm.models.base import create_attention_mask
            mask = create_attention_mask(h, cache[0])

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)

        if self.end_layer == self.num_hidden_layers:
            h = self.norm(h)
        return h


class Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.args = config
        self.model_type = config.model_type
        self.start_layer = config.start_layer
        self.end_layer = config.end_layer
        self.model = LanguageModel(config)
        
        if self.end_layer == self.args.num_hidden_layers:
            if config.tie_word_embeddings:
                self.lm_head = None
            else:
                self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ):
        out = self.model(inputs, cache)
        if self.end_layer == self.args.num_hidden_layers:
            if self.lm_head is not None:
                return self.lm_head(out)
            else:
                # Use tied embeddings
                return self.model.embed_tokens.as_linear(out)
        return out

    def sanitize(self, weights):
        total_layers = self.args.num_hidden_layers
        shard_state_dict = {}
        
        for key, value in weights.items():
            if key.startswith('model.layers.'):
                layer_num = int(key.split('.')[2])
                if self.start_layer <= layer_num < self.end_layer:
                    shard_state_dict[key] = value
            elif self.start_layer == 0 and key.startswith('model.embed_tokens'):
                shard_state_dict[key] = value
            elif self.end_layer == total_layers and (key.startswith('model.norm') or key.startswith('lm_head')):
                shard_state_dict[key] = value

        # Stack experts for MoE layers
        for l in range(self.args.num_hidden_layers):
            if self.start_layer <= l < self.end_layer:
                prefix = f"model.layers.{l}"
                # Check if this layer has MoE (not in mlp_only_layers)
                if l not in self.args.mlp_only_layers:
                    for n, m in [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")]:
                        for k in ["weight", "scales", "biases"]:
                            if f"{prefix}.mlp.experts.0.{m}.{k}" in shard_state_dict:
                                to_join = [
                                    shard_state_dict.pop(f"{prefix}.mlp.experts.{e}.{m}.{k}")
                                    for e in range(self.args.num_experts)
                                ]
                                shard_state_dict[f"{prefix}.mlp.switch_mlp.{m}.{k}"] = mx.stack(to_join)
        
        return shard_state_dict

    def make_cache(self):
        from mlx_lm.models.cache import KVCache
        return [KVCache() for _ in range(len(self.model.layers))]

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return self.args.head_dim

    @property
    def n_kv_heads(self):
        return self.args.num_key_value_heads
