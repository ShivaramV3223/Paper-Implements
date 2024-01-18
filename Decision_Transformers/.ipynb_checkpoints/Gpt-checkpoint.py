# Importing the libraries
import torch
import math
import numpy as np
from torch import nn, optim
import torch.nn.functional as F

# Class for configuring Attention Layer
class AttentionConfig:
    n_head = 8
    n_layers = 6
    embed_dim = 16
    attn_pdrop  = 0.1
    resid_pdrop = 0.1
    embd_pdrop = 0.1
    
    def __init__(self, vocab_size, block_size, model_type, max_timestep):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.model_type = model_type
        self.max_timestep = max_timestep

        
# Causal Self Attention Layer
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        #Layers for Calculating Keys, Queries and Value
        self.Key = nn.Linear(config.embed_dim, config.embed_dim, bias = False)
        self.Query = nn.Linear(config.embed_dim, config.embed_dim, bias = False)
        self.Value = nn.Linear(config.embed_dim, config.embed_dim, bias = False)
        
        # Dropout Layers for Regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop) # Attention layer
        self.resid_drop = nn.Dropout(config.resid_pdrop) # Residual layer
        
        # Projection layer
        self.projection = nn.Linear(config.embed_dim, config.embed_dim)
        
        self.mask = torch.tril(torch.ones(config.block_size + 1, config.block_size + 1).view(1, 1, config.block_size + 1, config.block_size + 1))
    
    # Forward function for the Attention transformer
    def forward(self, X):
        batch_size, seq_len, embed_dim = X.shape
        
        k = self.Key(X).view(batch_size, seq_len, self.config.n_heads, embed_dim // self.config.n_heads).transpose(1, 2) #(batch_size,n_heads, seq_len, head_dim)
        q = self.Query(X).view(batch_size, seq_len, self.config.n_heads, embed_dim // self.config.n_heads).transpose(1,2) # (batch_size, n_heads, seq_len, head_dim)
        v = self.Value(X).view(batch_size, seq_len, self.config.n_heads, embed_dim // self.config.n_heads).transpose(1,2) 
        
        attn = torch.einsum("nhsd,nhsd->nhss", [q,k]) #(batch_size, n_heads, seq_len, dq) x (batch_size, n_heads, seq_len, dk) -> (batch_size, n_heads, dq, dk)
        attn /= math.sqrt(self.config.dk)
        attn = attn.masked_fill(self.mask[:, :, :seq_len, :seq_len] == 0, float('-inf'))
        attn = F.softmax(attn, dim = -1)
        attn = self.attn_drop(attn)
        
        y = torch.einsum("nhss,nhsd -> nhsd", [attn, v])
        y = y.transpose(1,2).contiguous().view(batch_size, seq_len, embed_dim)
        
        y = self.resid_drop(self.projection(y))
        return y

# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.norm2 = nn.LayerNorm(config.embed_dim)
        self.attention = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
        nn.Linear(config.embed_dim, 4* config.embed_dim),
        nn.GELU(),
        nn.Linear(4 * config.embed_dim, config.embed_dim),
        nn.Dropout(config.resid_pdrop))
    
    def forward(self, X):
        X = X + self.attn(self.norm1(X))
        X = X + self.mlp(self.norm2(X))
        return X

# Complete GPT
class GPT(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        #Token Embedding
        self.tok_emb = nn.Embedding(config.vocab_size, config.embed_dim)
        # Positional Embedding
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size + 1, config.embed_dim))
        self.global_pos_emb = nn.Parameter(torch.zeros(1, config.max_timestep+1, config.embed_dim))
        
        self.drop = nn.Dropout(config.embd_pdrop)
        
        #Transformer
        self.transformer_blocks = nn.Sequential(*[TransformerBlock(config) for i in range(config.n_layers)])
        
        self.norm = nn.LayerNorm(config.embed_dim)
        self.head = nn.Linear(config.embed_dim, config.vocab_size, bias = False)
        
        # State Embedding
        self.state_encoder = nn.Sequential(nn.Linear(4, 16),
                                          nn.ReLU(),
                                          nn.Linear(16, 16),
                                          nn.ReLU(),
                                          nn.Linear(16, config.embed_dim),
                                          nn.Tanh())
        
        # Returns embeddings
        self.ret_emb = nn.Sequential(nn.Linear(1, config.embed_dim), nn.Tanh())
        
        self.apply(self._init_weights)
        
        # Actions embeddings
        self.action_embeddings = nn.Sequential(nn.Embedding(config.vocab_size, config.embed_dim), nn.Tanh())
        nn.init.normal_(self.action_embeddings[0].weight, mean=0.0, std=0.02)
        
        
    # Function to initialize the weights
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear):
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1)
            module.bias.data.zero_()
    
    # Optimizer settings for different parameters
    def configure_optimizers(self, train_config):
        decay = set()
        nodecay = set()
        
        decay_modules = (torch.nn.Linear)
        nodecay_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        
        for mod_name, module in self.named_modules():
            for param_name, parameter in module.named_parameters():
                full_param_name = '%s.%s' %(mod_name,param_name) if mod_name else param_name
                
                if param_name.endswith("bias"):
                    no_decay.add(full_param_name)
                elif param_name.endswith('weight') and isinstance(module, decay_modules):
                    decay.add(full_param_name)
                elif param_name.endswith("weight") and isinstance(module, no_decay_modules):
                    no_decay.add(full_param_name)
        
        no_decay.add('pos_emb')
        no_decay.add('global_pos_emb')
        
        param_dict = {pn:p for pn,p in self.named_parameters()}
        optim_grps = [
            {"params":[param_dict[pn] for pn in sorted(list(decay))], "weight_decay":self.config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        
        optimizer = optim.AdamW(optim_grps, lr = self.config.learning_rate, betas = self.config.betas)
        return optimizer
    
    
    # Forward function
    def forward(self, states, actions, targets =None, rtgs = None, timesteps = None):
        # states: (batch, block_size, 4)
        # actions: (batch, blocksize, 1)
        # targets: (batch, blocksize, 1)
        # rtgs: (batch, blocksize, 1)
        # timesteps: (batch, 1, 1)
        
        
        # State Embeddings 
        state_embeddings = self.state_encoder(states)
        
        # Creating Token Embedding according to the model configuration
        if actions is not None  and self.config.model_type == " reward_conditioned":
            rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))
            action_embeddings = self.action_embeddings(actions.type(torch.long).squeeze(-1))
            
            token_embeddings = torch.zeros((states.shape[0], states.shape[1]*3, self.config.embed_dim), dtype = torch.float32)
            token_embeddings[:, ::3, :] = rtg_embeddings
            token_embeddings[:,1::3, :] = state_embeddings
            token_embeddings[:,2::3, :] = action_embeddings[:, -states.shape[1] + int(targets is None): ]
            
        elif actions is None and self.config.model_type == "reward_conditioned":
            rtg_embeddings = self,ret_emb(rtgs.type(torch.float32))
            
            token_embeddings = torch.zeros((states.shape[0], states.shape[1] * 2, self.config.embed_dim), dtype = torch.float32)
            token_embeddings[:, ::2, :] = rtg_embeddings
            token_embeddings[:,1::2, :] = state_embeddings
            
        elif actions is not None and self.config.model_type == 'naive':
            action_embeddings = self.action_embeddings(actions.type(torch.long).squeeze(-1))
            
            token_embeddings = torch.zeros((states.shape[0], states.shape[1] * 2, self.config.embed_dim),
                                          dtype = torch.float32)
            token_embeddings[:,  ::2, :] = state_embeddings
            token_embeddings[:, 1::2, :] = action_embeddings[:, -states.shape[1] + int(targets is None):, :]
        elif actions is None and self.config.model_type == "Naive":
            token_embeddings = state_embeddings
        
        batch_size = states.shape[0]
        all_global_pos_emb = torch.repeat_interleave(self.global_pos_emb, batch_size, dim = 0)
        position_embeddings = torch.gather(all_global_pos_emb, 1, torch.repeat_interleave(timesteps, self.config.embed_dim, dim = 1)) + self.pos_emb[:, :token_embeddings.shape[1], :]
        
        x = self.drop(token_embeddings + position_embeddings)
        x = self.transfomer_blocks(x)
        x = self.norm(x)
        logits = self.head(x)
        
        if actions is not None and self.model_type == 'reward_conditioned':
            logits = logits[:, 1::3, :]
        elif actions is None and self.model_type == 'reward_conditioned':
            logits = logits[: ,1:,:]
        elif actions is not None and self.model_type == 'naive':
            logits = logits[: ,::2, :]
        elif actions is None and self.model_type == "naive":
            logtis = logits
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        
        return logits, loss
    