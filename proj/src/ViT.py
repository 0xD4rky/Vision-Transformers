from base import *

print("hi")

class PatchEmbeddings(nn.Module):
    """
    Convert the image into patches and then project them into a vector space.
    """

    def __init__(self, config):
        super().__init__()
        self.image_size = config["image_size"]
        self.patch_size = config["patch_size"]
        self.num_channels = config["num_channels"]
        self.hidden_size = config["hidden_size"]
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.projection = nn.Conv2d(self.num_channels, self.hidden_size, kernel_size=self.patch_size, stride=self.patch_size)

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
        
        # create learnable positional encoding and add +1 dim for [CLS]
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
    Attention module

    Will be used in:
        Multi-headed-attention Module
    """
    def __init__(self,vector_dim,attention_head_size,dropout,bias = True):
        
        super().__init__()
        self.vector_dim = vector_dim
        self.attention_head_size = attention_head_size
        self.dropout = nn.Dropout(dropout)
        
        # {query,key,value}
        self.query = nn.Linear(vector_dim,attention_head_size, bias = bias)
        self.key = nn.Linear(vector_dim, attention_head_size,bias = bias)
        self.value = nn.Linear(vector_dim,attention_head_size,bias = bias)
        
    def forward(self,x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        # i have them in matrix form
        
        similarity = torch.matmul(query,key.transpose(-1,-2))
        attention_probs = nn.functional.softmax((similarity/math.sqrt(self.attention_head_size)),dim = 1)
        attention_probs = self.dropout(attention_probs)
        output = torch.matmul(attention_probs,value)
        return output,attention_probs
        
class MultiheadAttention(nn.Module):
    """
    Multi-headed-attention module

    Will be used in:
        Transformer Encoder
    """
    
    def __init__(self,config):
        super().__init()
        self.vector_dim = config["vector_dim"]
        self.num_attention_heads = config["num_attention_heads"]
        
        self.attention_head_size =  self.vector_sim // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.qkv_bias = config["qkv_bias"]
        #creating a list of attention heads
        self.heads = nn.ModuleList([])
        for _ in range(self.num_attention_heads):
            head = Attention(
                self.vector_dim,
                self.attention_head_size,
                config["attention_probs_dropout_prob"],
                self.qkv_bias
                )
            self.heads.append(head)
        
        # project attention output back to vector dim
        self.output_projection = nn.Linear(self.all_head_size,self.vector_dim)
        self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])
        
    def forward(self,x,output_attentions = False):
        attention_outputs = [head(x) for head in self.heads] # for each attention head
        attention_output = torch.cat([attention_output for attention_output, _ in attention_outputs],dim=-1)
        # Project the concatenated attention output back to the hidden size
        attention_output = self.output_projection(attention_output)
        attention_output = self.output_dropout(attention_output)
        # Return the attention output and the attention probabilities (optional)
        if not output_attentions:
            return (attention_output, None)
        else:
            attention_probs = torch.stack([attention_probs for _, attention_probs in attention_outputs], dim=1)
            return (attention_output, attention_probs)

class MLP(nn.Module):
    """
    Multi-Layer Perceptron Module
    """
    
    def __init__(self,config):
        super().__init__()
        self.dense_1 = nn.Linear(config["vector_dim"],config["hidden_size"])
        self.act = NewGELUActivation()
        self.dense_2 = nn.Linear(config["hidden_size"],config["vector_dim"])
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])
        
    def forward(self,x):
        x = self.dense_1(x)
        x = self.act(x)
        x = self.dense_2(x)
        x = self.dropout(x)
        return x
    
class Block(nn.Module):
    
    """
    Single transformer block
    """
    
    def __init__(self,config):
        super().__init__()
        self.attention = MultiheadAttention(config)
        self.layer_norm1 = nn.LayerNorm(config["vector_dim"])
        self.mlp = MLP(config)
        self.layernorm_2 = nn.LayerNorm(config["vector_dim"])
        
    def forward(self,x,output_attentions = False):
        # {self-attention after normalizing layers}
        attention_output, attention_prob = self.attention(self.layer_norm1(x),output_attentions=output_attentions)
        x = x + attention_output # {skip-connections}\
        mlp_output = self.mlp(self.layer_norm2(x)) #{ffn}
        x = x + mlp_output
        if not output_attentions:
            return (x,None)
        else: 
            return (x,attention_prob)

class Encoder(nn.Module):
    
    def __init__(self,config):
        super().__init__()
        
        self.blocks = nn.ModuleList([])
        for _ in range(config["num_hidden_layers"]):
            block = Block(config)
            self.blocks.append(block)

    def forward(self, x, output_attentions=False):
        # Calculate the transformer block's output for each block
        all_attentions = []
        for block in self.blocks:
            x, attention_probs = block(x, output_attentions=output_attentions)
            if output_attentions:
                all_attentions.append(attention_probs)
        # Return the encoder's output and the attention probabilities (optional)
        if not output_attentions:
            return (x, None)
        else:
            return (x, all_attentions)

class Classification(nn.Module):
    
    """
    ViT model for classification
    """       
        
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.img_size = config["img_size"]
        self.vector_dim = config["vector_dim"]
        self.num_classes = config["num_classes"]
        
        # follow the below pipepline :)
        
        self.embeddings = Embeddings(config)
        self.encoder = Encoder(config)
        self.classifier = nn.Linear(self.vector_dim,self.num_classes)
        self.apply(self._init_weights)
        
    def forward(self, x, output_attentions=False):
        # Calculate the embedding output
        embedding_output = self.embedding(x)
        # Calculate the encoder's output
        encoder_output, all_attentions = self.encoder(embedding_output, output_attentions=output_attentions)
        # Calculate the logits, take the [CLS] token's output as features for classification
        logits = self.classifier(encoder_output[:, 0, :])
        # Return the logits and the attention probabilities (optional)
        if not output_attentions:
            return (logits, None)
        else:
            return (logits, all_attentions)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config["initializer_range"])
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


print("clear hai boss")
            
            
        