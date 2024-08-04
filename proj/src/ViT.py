from base import *

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
        Single head attention
        Will be used in Multi Heads
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
    

        
        
            
            
            
        