import torch
from attention import causal_attention

def mha_forward_gpt2(
    q, k, v,
    attention_mask=None,
    layer_past=None,
    use_cache=False,
):
    # q,k,v: [B, H, S, D]

    if layer_past is not None:
        past_k, past_v = layer_past
        k = torch.cat([past_k, k], dim=2)
        v = torch.cat([past_v, v], dim=2)

    out = causal_attention(q, k, v)

    new_past = (k, v) if use_cache else None
    return out, new_past
