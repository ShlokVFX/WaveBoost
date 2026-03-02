import torch
import torch.nn as nn
import math

def attention(q , k , v):
    #q , k , v : BHSD
    d_k = q.shape[-1] # D
    #BHSD x BHDS -> BHSS
    scores = torch.matmul(q , k.transpose(-2 , -1)) / math.sqrt(d_k)
    #BHSS -> BHSS
    weights = torch.softmax(scores , dim = -1)
    #BHSS x BHSD -> BHSD
    return torch.matmul(weights , v)


def causal_attention(q , k , v): #BHSD
    d_k = q.shape[-1]
    scores = torch.matmul(q , k.transpose(-2 , -1)) / math.sqrt(d_k)
    ## SS upper Triangular mask(Future)
    mask = torch.triu(torch.ones(scores.shape[-1] , scores.shape[-1]) , diagonal = 1).bool().to(scores.device)

    #BHSS (Masked)
    scores = scores.masked_fill(mask , float('-inf'))
    #BHSS -> BHSS
    weights = torch.softmax(scores , dim=-1)
    #BHSS x BHSD -> BHSD
    return torch.matmul(weights , v)

#MHA
class MultiHeadAttention(nn.Module):
    def __init__ (self , num_heads , hidden_dim):
        super().__init__()

        assert hidden_dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim , hidden_dim)
        self.k_proj = nn.Linear(hidden_dim , hidden_dim)
        self.v_proj= nn.Linear(hidden_dim , hidden_dim)
        self.out_proj = nn.Linear(hidden_dim , hidden_dim)

    def forward(self , x):
        B , S , _ = x.shape # B = Batch , S = Seq , H*D = hidden

        #BSH -> BSHD -> BHSD
        q = self.q_proj(x).view(B , S , self.num_heads , self.head_dim).transpose(1,2) #
        k = self.k_proj(x).view(B , S , self.num_heads , self.head_dim).transpose(1,2)
        v = self.v_proj(x).view(B , S , self.num_heads , self.head_dim).transpose(1,2)

        #BHSD x BHDS -> BHSS -> BHSD
        out = causal_attention(q , k , v)
        #BHSD -> BSHD -> BSH
        out = out.transpose(1,2).contiguous().view(B , S, -1)
        #BSH -> BSH 
        return self.out_proj(out) 

#GQA
class GroupedQueryAttention(nn.Module):
    def __init__ (self ,hidden_dim , num_q_heads , num_kv_heads):
        super().__init__()
        #B = batch
        #S = sequence length
        #H = total query heads
        #KVH = key/value heads
        #D = head_dim

        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.num_groups = num_q_heads // num_kv_heads
        self.head_dim = hidden_dim // num_q_heads

        assert hidden_dim % num_q_heads == 0
        assert hidden_dim % num_kv_heads == 0

        #Q: BSH -> BS(QH*D)
        self.q_proj = nn.Linear(hidden_dim , num_q_heads * self.head_dim)
        #K,V: BSH -> BS(KVH*D)
        self.k_proj = nn.Linear(hidden_dim , num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(hidden_dim , num_kv_heads* self.head_dim)

        self.out_proj = nn.Linear(hidden_dim , hidden_dim)

    def forward(self , x):
        B , S, _ = x.shape

        #BSH -> BSQHD -> BQHSD
        q = self.q_proj(x).view(B , S , self.num_q_heads , self.head_dim).transpose(1,2)
        #BSH -> BSKVHD -> BKVHSD
        k = self.k_proj(x).view(B , S , self.num_kv_heads , self.head_dim).transpose(1,2)
        v = self.v_proj(x).view(B , S , self.num_kv_heads , self.head_dim).transpose(1,2)
        #BKVHSD -> BQHSD (repeat KV per group)
        k = k.repeat_interleave(self.num_groups , dim =1)
        v = v.repeat_interleave(self.num_groups , dim =1) #dim=2 would repeat sequence length

        #BQHSD X BQHDS -> BQHSS -> BQHSD
        out = causal_attention(q , k , v)

        #BQHSD -> BSHD -> BSH
        out = out.transpose(1,2).contiguous().view(B,S,-1)

        return self.out_proj(out)
    
# MLA
class MultiLatentAttention(nn.Module):
    def __init__(self, hidden_dim, num_q_heads, latent_dim):
        super().__init__()

        assert hidden_dim % num_q_heads == 0

        self.num_q_heads = num_q_heads
        self.head_dim = hidden_dim // num_q_heads
        self.latent_dim = latent_dim

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)

        # K/V compressed into latent space
        self.k_latent = nn.Linear(hidden_dim, latent_dim, bias=False)
        self.v_latent = nn.Linear(hidden_dim, latent_dim, bias=False)

        # Latent → per-head K/V
        self.k_proj = nn.Linear(latent_dim, num_q_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(latent_dim, num_q_heads * self.head_dim, bias=False)

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        B, S, _ = x.shape

        # Q: BSH → BQHSD
        q = self.q_proj(x).view(B, S, self.num_q_heads, self.head_dim).transpose(1, 2)

        # Latent K/V: BSH → BSR
        k_lat = self.k_latent(x)
        v_lat = self.v_latent(x)

        # Expand latent → per-head K/V
        # BSR → BS(QH*D) → BQHSD
        k = self.k_proj(k_lat).view(B, S, self.num_q_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(v_lat).view(B, S, self.num_q_heads, self.head_dim).transpose(1, 2)

        # Attention
        out = causal_attention(q, k, v)

        # BQHSD → BSH
        out = out.transpose(1, 2).contiguous().view(B, S, -1)

        return self.out_proj(out)
