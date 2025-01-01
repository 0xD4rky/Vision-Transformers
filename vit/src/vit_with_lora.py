import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from base import *

class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer"""
    def __init__(self, in_features, out_features, rank=4, alpha=16):
        super(LoRALayer,self).__init__()
        self.rank = rank
        self.scaling = alpha / rank
        
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        x = x @ (self.lora_A @ self.lora_B) * self.scaling
        return x
    
class PatchEmbeddings(nn.Module):
    """
    Convert the image into patches and then project them into a vector space.
    """

    def __init__(self, config):
        super().__init__()
        self.image_size = config["image_size"]
        self.patch_size = config["patch_size"]
        self.num_channels = config["num_channels"]
        self.vector_dim = config["vector_dim"]
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.projection = nn.Conv2d(self.num_channels, self.vector_dim, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x):
        # {batch_size, num_channels, image_size, image_size}-> {batch_size, num_patches, vector_dim}
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class Embeddings(nn.Module):
    
    """
    adding positional information to extracted patch embeddings
    """
    
    def __init__(self,config):
        self.config = config
        self.patch_emb = PatchEmbeddings(config)
        self.cls_token = nn.Parameter(torch.randn(1,1,config["vector_dim"]))
        self.positional_encoding = nn.Parameter(torch.randn(1,self.patch_emb.num_patches + 1, config["vector_dim"]))
        self.droput = nn.Dropout(config["droput_prob"])
        
    def forward(self,x):
        x = self.patch_emb(x)
        batch_size, _, _ = x.size()
        # expand the [cls] token to batch size
        #{1,1,vector_dim} -> (batch_size,1,hidden_size)
        cls_tokens = self.cls_token.expand(batch_size,-1,-1)
        """
        concatenating cls token to inputn sequence
        size : {num_patches + 1}
        """
        x = torch.cat((cls_tokens,x),dim = 1)
        x = x + self.positional_encoding
        return x
    
class Attention(nn.Module):
    """
    Attention module with LoRA Support
    """
    def __init__(self,vector_dim,attention_head_size,dropout,bias=True, use_lora=False, lora_rank=8, lora_alpha=16):
        super().__init__()
        self.vector_dim = vector_dim
        self.attention_head_size = attention_head_size
        self.dropout = nn.Dropout(dropout)
        self.use_lora = use_lora
        self.query = nn.Linear(vector_dim, attention_head_size, bias = bias)
        self.key = nn.Linear(vector_dim, attention_head_size, bias = bias)
        self.value = nn.Linear(vector_dim, attention_head_size, bias = bias)

        if use_lora:
            self.lora_q = LoRALayer(vector_dim, attention_head_size, lora_rank, lora_alpha)
            self.lora_v = LoRALayer(vector_dim, attention_head_size, lora_rank, lora_alpha)
        
    def forward(self, x):

        q = self.query(x)
        key = self.key(x)
        v = self.value(x)

        if self.use_lora:
            query = q + self.lora_q(x)
            value = v + self.lora_v(x)
        
        similarity = torch.matmul(query, key.transpose(-1,-2))
        attention_probs = F.softmax((similarity/math.sqrt(self.attention_head_size)),dim = 1)
        attention_probs = self.dropout(attention_probs)
        output = torch.matmul(attention_probs, value)
        return output, attention_probs
    
class MultiheadAttention(nn.Module):
    """
    Multi-headed-attention module with LoRA support
    """
    def __init__(self, config):
        super().__init__()
        self.vector_dim = config["vector_dim"]
        self.num_attention_heads = config["num_attention_heads"]
        
        self.attention_head_size = self.vector_dim // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.qkv_bias = config["qkv_bias"]
        self.use_lora = config.get("use_lora", False)  # New parameter
        self.lora_rank = config.get("lora_rank", 8)    # New parameter
        self.lora_alpha = config.get("lora_alpha", 16) # New parameter
        
        self.heads = nn.ModuleList([
            Attention(
                self.vector_dim,
                self.attention_head_size,
                config["attention_probs_dropout_prob"],
                self.qkv_bias,
                self.use_lora,
                self.lora_rank,
                self.lora_alpha
            )
            for _ in range(self.num_attention_heads)
        ])
        
        self.output_projection = nn.Linear(self.all_head_size, self.vector_dim)
        self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])
    
    def forward(self, x, output_attentions=False):
        attention_outputs = [head(x) for head in self.heads]
        attention_output = torch.cat(
            [attention_output for attention_output, _ in attention_outputs],
            dim=-1
        )
                
        attention_output = self.output_projection(attention_output)
        attention_output = self.output_dropout(attention_output)
        
        if not output_attentions:
            return (attention_output, None)
        
        attention_probs = torch.stack(
            [attention_probs for _, attention_probs in attention_outputs],
            dim=1
        )
        return (attention_output, attention_probs)

class MLP(nn.Module):
    """
    Multi-Layer Perceptron Module with LoRA support
    """
    
    def __init__(self, config):
        super().__init__()
        self.use_lora = config.get("use_lora", False)
        self.lora_rank = config.get("lora_rank", 8)
        self.lora_alpha = config.get("lora_alpha", 16)
        self.dense_1 = nn.Linear(config["vector_dim"], config["hidden_size"])
        self.dense_2 = nn.Linear(config["hidden_size"], config["vector_dim"])
        
        if self.use_lora:
            self.lora_1 = LoRALayer(
                config["vector_dim"],
                config["hidden_size"],
                self.lora_rank,
                self.lora_alpha
            )
            self.lora_2 = LoRALayer(
                config["hidden_size"],
                config["vector_dim"],
                self.lora_rank,
                self.lora_alpha
            )
        
        self.act = NewGELUActivation()
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])
        
    def forward(self, x):
        hidden = self.dense_1(x)
        if self.use_lora:
            hidden = hidden + self.lora_1(x)
        hidden = self.act(hidden)
        output = self.dense_2(hidden)
        if self.use_lora:
            output = output + self.lora_2(hidden)
        output = self.dropout(output)
        return output
