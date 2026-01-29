def attention(q , k , v):
    d_k = q.shape[-1]
    scores = torch.matmul(q , k.transpose[-2 , -1]) / math.sqrt(d_k)
    weights = torch.softmax(scores , dim = -1)
    return torch.matmul(weights , v)


def causal_attention(q , k , v):
    d_k = q.shape[-1]
    scores = torch.matmul(q , k.transpose(-2 , -1)) / math.sqrt(d_k)
    mask = torch.triu(torch.ones(scores.shape[-1] , scores.shape[-1]) , diagonal = 1).bool()
    scores = scores.masked_fill(mask , float('-inf'))
    weights = torch.softmax(scores , dim=-1)
    return torch.matmul(weights , v)

class MultiHeadAttention(nn.Module):
    
    def __init__ (self , num_heads , hidden_dim):
        self.num_heads , self_head_dim = num_heads , hidden_dim // num_heads
        self.q_proj = nn.Linear(hidden_dim , hidden_dim)
        self.k_proj = nn.Linear(hidden_dim , hidden_dim)
        self.v_proj = nn.Linear(hidden_dim , hidden_dim)
        self.out_proj = nn.Linear(hidden_dim , hidden_dim)

