import os
import re
import time
import types
import torch
import numpy as np
import matplotlib.pyplot as plt

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate.utils import (
    convert_model,
    contextual_fp8_autocast,
    has_transformer_engine_layers,
)
from transformer_engine.common import recipe as te_recipe
from transformer_engine.common.recipe import Format


# =========================================
# Config
# =========================================

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
MAX_NEW_TOKENS = 1

torch.manual_seed(42)
np.random.seed(42)


# =========================================
# Dataset Utilities
# =========================================

def build_samples(ds):
    samples = []
    for item in ds:
        words = item["text"].split()
        if len(words) < 2:
            continue
        prefix = " ".join(words[:-1])
        target = re.sub(r"[^\w'-]+$", "", words[-1]).lower()
        samples.append((prefix, target))
    return samples


def batch_iter(lst, bs):
    for i in range(0, len(lst), bs):
        yield lst[i:i + bs]


def normalize(word):
    return re.sub(r"[^\w'-]+$", "", word).lower()


# =========================================
# Evaluation
# =========================================

@torch.inference_mode()
def evaluate_model(model, tokenizer, test_dataset,
                   use_te_fp8=False, fp8_recipe=None):
    samples = build_samples(test_dataset)
    model.eval()

    total_time = 0.0
    correct = 0
    total = 0

    for batch in batch_iter(samples, BATCH_SIZE):
        prefixes = [p for p, _ in batch]
        targets = [t for _, t in batch]

        inputs = tokenizer(
            prefixes,
            return_tensors="pt",
            padding=True,
            truncation=True,
            pad_to_multiple_of=16,
        ).to(model.device)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()

        if use_te_fp8 and fp8_recipe is not None:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
        else:
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        total_time += time.perf_counter() - start

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for pred_text, tgt in zip(decoded, targets):
            match = re.match(r"\s*([A-Za-z]+(?:['-][A-Za-z]+)?)", pred_text)
            pred = normalize(match.group(1)) if match else ""
            correct += int(pred == tgt)
            total += 1

    avg_time = total_time / total
    accuracy = correct / total

    return total_time, avg_time, accuracy


# =========================================
# FP6 (Your Custom Quantization)
# =========================================

def quantize_to_fp6(tensor):
    """
    Simplified FP6 quantization.
    6 bits → 64 discrete levels.
    Linear scaling approximation.
    """
    min_val = tensor.min()
    max_val = tensor.max()
    scale = (max_val - min_val) / 63
    quantized = ((tensor - min_val) / scale).round().clamp(0, 63)
    return quantized.float() / 63 * (max_val - min_val) + min_val


def apply_fp6_quantization(model):
    for param in model.parameters():
        param.data = quantize_to_fp6(param.data)


# =========================================
# Main Experiment
# =========================================

def run_experiment():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print("Loading dataset...")
    dataset = load_dataset("EleutherAI/lambada_openai")
    test_dataset = dataset["test"].select(range(128))

    results = []

    # =========================================
    # FP16
    # =========================================
    print("\nEvaluating FP16...")
    model_fp16 = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    t_fp16, avg_fp16, acc_fp16 = evaluate_model(
        model_fp16, tokenizer, test_dataset
    )

    results.append(("FP16", t_fp16, avg_fp16, acc_fp16))


    # =========================================
    # INT8 (bitsandbytes)
    # =========================================
    print("\nEvaluating INT8...")
    model_int8 = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        load_in_8bit=True,
        device_map="auto",
    )

    t_int8, avg_int8, acc_int8 = evaluate_model(
        model_int8, tokenizer, test_dataset
    )

    results.append(("INT8", t_int8, avg_int8, acc_int8))


    # =========================================
    # FP8 (TransformerEngine)
    # =========================================
    print("\nEvaluating FP8...")
    model_fp8 = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    convert_model(
        model_fp8,
        to_transformer_engine=True,
        _convert_linear=True,
        _convert_ln=True,
    )

    assert has_transformer_engine_layers(model_fp8)

    fp8_recipe = te_recipe.DelayedScaling(
        fp8_format=Format.HYBRID,
        amax_history_len=16,
        amax_compute_algo="max",
    )

    wrapped_forward = contextual_fp8_autocast(
        model_fp8.forward,
        fp8_recipe=fp8_recipe,
        use_during_eval=True,
    )

    model_fp8.forward = types.MethodType(wrapped_forward, model_fp8)

    t_fp8, avg_fp8, acc_fp8 = evaluate_model(
        model_fp8,
        tokenizer,
        test_dataset,
        use_te_fp8=True,
        fp8_recipe=fp8_recipe,
    )

    results.append(("FP8", t_fp8, avg_fp8, acc_fp8))


    # =========================================
    # FP6
    # =========================================
    print("\nEvaluating FP6 (Simulated)...")
    model_fp6 = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    apply_fp6_quantization(model_fp6)

    t_fp6, avg_fp6, acc_fp6 = evaluate_model(
        model_fp6,
        tokenizer,
        test_dataset,
    )

    results.append(("FP6", t_fp6, avg_fp6, acc_fp6))


    # =========================================
    # Plot
    # =========================================

    base_time = results[0][1]

    precisions = [r[0] for r in results]
    speedups = [base_time / r[1] for r in results]
    accuracies = [r[3] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.bar(precisions, speedups)
    ax1.set_title("Speedup vs FP16")
    ax1.set_ylabel("Speedup")
    ax1.grid(axis="y", linestyle="--", alpha=0.7)

    ax2.bar(precisions, accuracies)
    ax2.set_title("Accuracy Across Precisions")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1.0)
    ax2.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_experiment()