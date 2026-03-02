import torch 
import torch.nn as nn
import nn.functional as F
import math

#Attention(Q , K , V) = softmax((QK^T)/sqrt(dim_k))V
def sdpa(Q,K,V):
    d_k = Q.size()[-1]
    score=torch.mamtul(Q , K.transpose(-1 , -2)) //math.sqrt(d_k)
    