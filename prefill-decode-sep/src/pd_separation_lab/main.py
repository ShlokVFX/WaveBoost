import torch
import torch.nn as nn
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# =========================
# Transformer Layer
# =========================
class SimplifiedTransformerLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True,
        )

        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        attn_output, _ = self.self_attn(
            query=x,
            key=x if kv_cache is None else kv_cache[0],
            value=x if kv_cache is None else kv_cache[1],
            need_weights=False,
        )

        x = self.norm1(x + attn_output)
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)

        return x, (x, x)


# =========================
# Simple LM
# =========================
class SimpleLanguageModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)

        self.layers = nn.ModuleList(
            [
                SimplifiedTransformerLayer(hidden_size, num_heads)
                for _ in range(num_layers)
            ]
        )

        self.output_proj = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, kv_caches=None):
        x = self.embedding(input_ids)

        if kv_caches is None:
            kv_caches = [None] * len(self.layers)

        new_caches = []

        for i, layer in enumerate(self.layers):
            x, cache = layer(x, kv_caches[i])
            new_caches.append(cache)

        logits = self.output_proj(x)
        return logits, new_caches


# =========================
# Traditional Deployment
# =========================
class TraditionalDeployment:
    def __init__(self, model):
        self.model = model.to(device)
        self.model.eval()

    def process_request(self, prompt, output_len):
        prompt_tensor = torch.tensor([prompt], device=device)

        # Prefill
        start_prefill = time.time()
        with torch.no_grad():
            logits, kv = self.model(prompt_tensor)
        prefill_time = time.time() - start_prefill

        # Decode
        start_decode = time.time()
        generated = []
        current_token = torch.argmax(logits[:, -1, :], dim=-1)

        with torch.no_grad():
            for _ in range(output_len):
                logits, kv = self.model(current_token.unsqueeze(1), kv)
                next_token = torch.argmax(logits[:, -1, :], dim=-1)
                generated.append(next_token.item())
                current_token = next_token

        decode_time = time.time() - start_decode

        return generated, prefill_time + decode_time, prefill_time, decode_time


# =========================
# Prefill-Decode Separation
# =========================
class PDSeparationDeployment:
    def __init__(self, model):
        self.prefill_model = model.to(device)
        self.decode_model = model.to(device)
        self.prefill_model.eval()
        self.decode_model.eval()

    def process_request(self, prompt, output_len):
        prompt_tensor = torch.tensor([prompt], device=device)

        # Prefill
        start_prefill = time.time()
        with torch.no_grad():
            logits, kv = self.prefill_model(prompt_tensor)
        prefill_time = time.time() - start_prefill

        # Simulate KV transfer
        start_transfer = time.time()
        transferred_kv = [(k.clone(), v.clone()) for k, v in kv]
        transfer_time = time.time() - start_transfer

        # Decode
        start_decode = time.time()
        generated = []
        current_token = torch.argmax(logits[:, -1, :], dim=-1)

        with torch.no_grad():
            for _ in range(output_len):
                logits, transferred_kv = self.decode_model(
                    current_token.unsqueeze(1), transferred_kv
                )
                next_token = torch.argmax(logits[:, -1, :], dim=-1)
                generated.append(next_token.item())
                current_token = next_token

        decode_time = time.time() - start_decode

        total = prefill_time + transfer_time + decode_time
        return generated, total, prefill_time, decode_time, transfer_time


# =========================
# Workload Generator
# =========================
def generate_workload(n):
    workload = []
    for _ in range(n):
        prompt_len = np.random.randint(10, 100)
        output_len = np.random.randint(5, 30)
        prompt = np.random.randint(0, 10000, prompt_len).tolist()
        workload.append((prompt, output_len))
    return workload


# =========================
# Experiment
# =========================
def run_experiment():
    model = SimpleLanguageModel(
        vocab_size=10000,
        hidden_size=256,
        num_layers=4,
        num_heads=4,
    )

    workload = generate_workload(10)

    traditional = TraditionalDeployment(model)
    pd = PDSeparationDeployment(model)

    trad_times = []
    pd_times = []
    transfer_times = []

    print("\nRunning experiment...\n")

    for i, (prompt, out_len) in enumerate(workload):
        print(f"Processing request {i+1}/{len(workload)}")

        _, t_total, _, _ = traditional.process_request(prompt, out_len)
        _, p_total, _, _, transfer = pd.process_request(prompt, out_len)

        trad_times.append(t_total)
        pd_times.append(p_total)
        transfer_times.append(transfer)

    print("\n========== Results ==========")
    print(f"Avg Traditional: {np.mean(trad_times):.4f}s")
    print(f"Avg PD-Separated: {np.mean(pd_times):.4f}s")
    print(f"Avg KV Transfer: {np.mean(transfer_times):.4f}s")
    print("=============================\n")


if __name__ == "__main__":
    run_experiment()