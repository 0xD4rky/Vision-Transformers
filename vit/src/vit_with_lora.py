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
        self.use_lora = config.get("use_lora", False)  
        self.lora_rank = config.get("lora_rank", 8)    
        self.lora_alpha = config.get("lora_alpha", 16) 
        
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

def prepare_mlp_for_lora_training(model):
    """Freeze all parameters except LoRA parameters"""
    for name, param in model.named_parameters():
        if 'lora' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    return model

class Block(nn.Module):

    "single transformer block with LoRA support"

    def __init__(self, config):
        super().__init_()
        self.attention = MultiheadAttention(config)
        self.layer_norm1 = nn.LayerNorm(config["vector_dim"])
        self.mlp = MLP(config)
        self.layer_norm2 = nn.LayerNorm(config["vector_dim"])
    
    def forward(self, x, output_attentions = False):
        attention_output, attention_probs = self.attention(self.layer_norm1(x), output_attentions=output_attentions)
        x = x + attention_output
        mlp_output = self.mlp(self.layer_norm2(x))
        x = x + mlp_output
        if not output_attentions:
            return (x, None)
        else:
            return (x, attention_probs)
        
class Encoder(nn.Module):
    """
    Transformer encoder with LoRA support
    """
    
    def __init__(self, config):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(config) for _ in range(config["num_hidden_layers"])
        ])

    def forward(self, x, output_attentions=False):
        all_attentions = []
        
        for block in self.blocks:
            x, attention_probs = block(x, output_attentions=output_attentions)
            if output_attentions:
                all_attentions.append(attention_probs)
                
        if not output_attentions:
            return (x, None)
        else:
            return (x, all_attentions)
        
class LoRALinear(nn.Module):
    """
    Linear layer with LoRA support for classification head
    """
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.lora = LoRALayer(in_features, out_features, rank, alpha)
        
    def forward(self, x):
        return self.linear(x) + self.lora(x)
    
class Classification(nn.Module):
    """
    ViT model for classification with LoRA support
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.img_size = config["img_size"]
        self.vector_dim = config["vector_dim"]
        self.num_classes = config["num_classes"]
        
        # Initialize components
        self.embeddings = Embeddings(config)
        self.encoder = Encoder(config)
        
        # Use LoRA for classifier if enabled
        if config.get("use_lora", False):
            self.classifier = LoRALinear(
                self.vector_dim,
                self.num_classes,
                config.get("lora_rank", 8),
                config.get("lora_alpha", 16)
            )
        else:
            self.classifier = nn.Linear(self.vector_dim, self.num_classes)
            
        self.apply(self._init_weights)
        
    def forward(self, x, output_attentions=False):
        embedding_output = self.embeddings(x)
        encoder_output, all_attentions = self.encoder(
            embedding_output,
            output_attentions=output_attentions
        )
        
        # Use CLS token for classification
        logits = self.classifier(encoder_output[:, 0, :])
        
        if not output_attentions:
            return (logits, None)
        else:
            return (logits, all_attentions)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            torch.nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config["initializer_range"]
            )
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, Embeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.config["initializer_range"],
            ).to(module.position_embeddings.dtype)

def prepare_model_for_lora_training(model):
    """
    Prepare the model for LoRA training by freezing non-LoRA parameters
    """
    for name, param in model.named_parameters():
        if 'lora' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    return model