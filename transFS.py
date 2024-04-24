import torch.nn as nn
import torch
from easydict import EasyDict as edict

configs = edict()

configs.hidden_dim = 4192
configs.vocab_size = 128_256

# https://huggingface.co/meta-llama/Meta-Llama-3-8B/blob/main/config.json


class LlamaRMSNorm(nn.module):
    def __init__(self, configs):
        super().__init__()

        ## People found re-centering of LayerNorm is non-essential
        self.weights = nn.Parameter(torch.ones(configs.hidden_dim))
        self.eps     = configs.eps

    def forward(self, x):
        # BatchSize, SeqLen, EmbDim
        var = torch.var(x, dim = -1, keepdim=True)
        x   = x/(torch.sqrt(var + self.eps))


        return x * self.weights # Element wise multiplication
    
class LlamaMLP(nn.module):
    def __init__(self, configs):
        super().__init__()
        self.gate_proj = nn.Linear(configs.hidden_dim, configs.intermediate_size, bias=False)
        self.up_proj   = nn.Linear(configs.hidden_dim, configs.intermediate_size, bias=False)
        self.down_proj = nn.Linear(configs.intermediate_size, configs.hidden_dim, bias=False)

    def forward(self, x):
        return self.down_proj(self.gate_proj(x)*self.up_proj(x))
    
    
class LlamaAttention(nn.module):
    def __init__(self, configs):
        super().__init__()
        
        ## Don't need all queries and values per column, can group up some of them to reduce space
        self.hidden_dim = configs.hidden_dim
        self.num_attention_heads = configs.num_attention_heads
        self.num_kv_heads   = configs.nu_kv_heads
        self.head_dim       = self.hidden_dim//self.num_attention_heads
        self.num_kv_heads   = self.
    def forward(self, x):
        return self.down_proj(self.gate_proj(x)*self.up_proj(x))
    

    
class LlamaDecoderLayer(nn.module):
    def __init__(self, configs):
        super().__init__()

        # Norms
        self.input_norm = LlamaRMSNorm(configs)
        self.output_norm = LlamaRMSNorm(configs)

        # Self Attention
        self.attn = LlamaAttention(configs)

        # MLP
        self.mlp = LlamaMLP(configs)


class LlamaModel(nn.Module):

    def __init__(self, configs):
        super().__init__()
        
        # Load the embeddings
        self.embeds = nn.Embedding() # initialised randomly, basically like a lookup table

        # RMS Norm instead of LayerNorm
        self.norm = LlamaRMSNorm(configs) ## Llama uses RMS Norm

         # Decoder Layers
        self.layers = nn.ModuleList([LlamaDecoderLayer(configs) for _ in range(configs.num_layers)]) ## Handles gradients vagaera on it's own
        

        # Final LM Head
        self.lm_head = nn.Linear() # Decode Token







