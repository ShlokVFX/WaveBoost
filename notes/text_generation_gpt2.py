import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# ---------------------------
# Config
# ---------------------------
MODEL_DIR = "./models/gpt2"
VIZ_DIR = "visualization"
os.makedirs(VIZ_DIR, exist_ok=True)

from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
from hf_attention_adapter import mha_forward_gpt2

def patched_gpt2_attention_forward(
    self,
    hidden_states,
    layer_past=None,
    attention_mask=None,
    use_cache=False,
    **kwargs,
):
    B, S, C = hidden_states.shape

    qkv = self.c_attn(hidden_states)
    q, k, v = qkv.split(self.split_size, dim=2)

    q = q.view(B, S, self.num_heads, -1).transpose(1, 2)
    k = k.view(B, S, self.num_heads, -1).transpose(1, 2)
    v = v.view(B, S, self.num_heads, -1).transpose(1, 2)

    attn_out, new_past = mha_forward_gpt2(
        q, k, v,
        attention_mask=attention_mask,
        layer_past=layer_past,
        use_cache=use_cache,
    )

    attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, C)
    attn_out = self.c_proj(attn_out)

    return attn_out, new_past

GPT2Attention.forward = patched_gpt2_attention_forward


# ---------------------------
# Load model & tokenizer
# ---------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)

# ---------------------------
# Single forward pass
# ---------------------------
prompt = "The quick brown fox jumped over the"
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits
last_logits = logits[0, -1, :]
next_token_id = last_logits.argmax()

# ---------------------------
# Autoregressive generation (no KV cache)
# ---------------------------
def generate_token(inputs):
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    return logits[0, -1, :].argmax()

generated_tokens = []
next_inputs = inputs
durations_s = []

for _ in range(10):
    t0 = time.time()
    next_token_id = generate_token(next_inputs)
    durations_s.append(time.time() - t0)

    next_inputs = {
        "input_ids": torch.cat(
            [next_inputs["input_ids"], next_token_id.reshape((1, 1))],
            dim=1),
        "attention_mask": torch.cat(
            [next_inputs["attention_mask"], torch.tensor([[1]])],
            dim=1),
    }

    generated_tokens.append(tokenizer.decode(next_token_id))

print("No KV cache total time:", sum(durations_s))
print(generated_tokens)

plt.figure()
plt.plot(durations_s)
plt.title("Token latency without KV cache")
plt.xlabel("Step")
plt.ylabel("Seconds")
plt.savefig(f"{VIZ_DIR}/no_kv_cache.png")
plt.close()

# ---------------------------
# Autoregressive generation (with KV cache)
# ---------------------------
def generate_token_with_past(inputs):
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    return logits[0, -1, :].argmax(), outputs.past_key_values

generated_tokens = []
next_inputs = inputs
durations_cached_s = []

for _ in range(10):
    t0 = time.time()
    next_token_id, past_key_values = generate_token_with_past(next_inputs)
    durations_cached_s.append(time.time() - t0)

    next_inputs = {
        "input_ids": next_token_id.reshape((1, 1)),
        "attention_mask": torch.cat(
            [next_inputs["attention_mask"], torch.tensor([[1]])],
            dim=1),
        "past_key_values": past_key_values,
    }

    generated_tokens.append(tokenizer.decode(next_token_id))

print("With KV cache total time:", sum(durations_cached_s))
print(generated_tokens)

plt.figure()
plt.plot(durations_s, label="No KV cache")
plt.plot(durations_cached_s, label="With KV cache")
plt.legend()
plt.title("Token latency comparison")
plt.xlabel("Step")
plt.ylabel("Seconds")
plt.savefig(f"{VIZ_DIR}/kv_cache_comparison.png")
plt.close()
