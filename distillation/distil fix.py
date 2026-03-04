# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.0",
#   "transformers>=4.40",
#   "huggingface_hub",
#   "datasets",
#   "numpy",
# ]
# ///

"""
CODE03: Large Model Qwen3 Distillation
======================================
Authors: Wangyuanshuo, ZOMI

Knowledge Distillation is a technique that transfers knowledge from a large
Teacher Model to a smaller Student Model, enabling the student to achieve
performance close to the teacher with far fewer parameters.

This experiment uses Qwen3-4B as the teacher and Qwen3-0.6B as the student.
Through distillation, we aim to bring the student's performance (on tasks like
math reasoning or code generation) close to the teacher's, while keeping its
parameter count and compute cost low.

Run with:
    uv run qwen3_distillation.py
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.amp import autocast, GradScaler
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# ─────────────────────────────────────────────
# 0. GPU Configuration
# ─────────────────────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,7"


# ─────────────────────────────────────────────
# 1. Core Idea: Distillation Loss Functions
# ─────────────────────────────────────────────

class DistillationLoss(nn.Module):
    """
    Classic soft-label + hard-label mixed distillation loss.

    Combines:
      - KL divergence loss (soft labels from teacher)
      - Cross-entropy loss (hard labels from ground truth)

    Total loss: L = alpha * L_KL + (1 - alpha) * L_CE

    Args:
        alpha (float): Weight for distillation loss vs. cross-entropy loss.
        temperature (float): Softmax temperature T. Higher T smooths the
                             probability distribution, revealing more
                             inter-class relationships ("dark knowledge").
    """

    def __init__(self, alpha: float = 0.7, temperature: float = 5.0):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, labels):
        # Soft teacher distribution (temperature-scaled)
        soft_teacher = torch.softmax(teacher_logits / self.temperature, dim=-1)
        # Log-softmax of student (required by KLDivLoss)
        log_soft_student = torch.log_softmax(student_logits / self.temperature, dim=-1)
        # KL divergence loss, scaled back by T^2 (standard distillation scaling)
        kl_loss = self.kl_loss(log_soft_student, soft_teacher) * (self.temperature ** 2)

        # Hard-label cross-entropy
        ce_loss = self.ce_loss(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1)
        )

        return self.alpha * kl_loss + (1 - self.alpha) * ce_loss


class DistillationLossWithChunk(nn.Module):
    """
    Chunk-based distillation loss to reduce peak memory usage.

    Splits the vocab dimension into `num_chunks` chunks and computes KL
    independently per chunk, then aggregates with a size-weighted average.

    ⚠ Limitation: Splitting softmax across chunks introduces bias because
    the normalization denominator is local to each chunk rather than the
    full vocabulary.

    Args:
        alpha (float): KL vs CE weight.
        temperature (float): Softmax temperature.
        pad_id (int, optional): Padding token ID to ignore in CE loss.
        num_chunks (int): Number of vocab chunks.
    """

    def __init__(
        self,
        alpha: float = 0.7,
        temperature: float = 5.0,
        pad_id: int = None,
        num_chunks: int = 4,
    ):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.num_chunks = num_chunks
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.ce_loss = (
            nn.CrossEntropyLoss(ignore_index=pad_id)
            if pad_id is not None
            else nn.CrossEntropyLoss()
        )

    def forward(self, student_logits, teacher_logits, labels):
        # Cast to float32 for numerical stability
        student_logits = student_logits.float()
        teacher_logits = teacher_logits.float()

        T = self.temperature
        vocab_size = student_logits.size(-1)

        # Split vocab dimension into chunks
        student_chunks = torch.chunk(student_logits, self.num_chunks, dim=-1)
        teacher_chunks = torch.chunk(teacher_logits, self.num_chunks, dim=-1)

        total_kl = 0.0
        for s_chunk, t_chunk in zip(student_chunks, teacher_chunks):
            soft_t = torch.softmax(t_chunk / T, dim=-1)
            log_s = torch.log_softmax(s_chunk / T, dim=-1)
            kl_chunk = self.kl_loss(log_s, soft_t) * (T * T)
            # Weight by relative chunk size
            total_kl += kl_chunk * (s_chunk.size(-1) / vocab_size)

        ce = self.ce_loss(student_logits.view(-1, vocab_size), labels.view(-1))
        return self.alpha * total_kl + (1.0 - self.alpha) * ce


class DistillationLossWithTopK(nn.Module):
    """
    Top-K truncated distillation loss (memory-efficient).

    Masks teacher logits to retain only the top-K highest-probability tokens,
    discarding the long tail. This significantly reduces memory pressure while
    preserving the most informative "dark knowledge".

    Reference: https://arxiv.org/html/2410.16215v1

    Args:
        alpha (float): KL vs CE weight.
        temperature (float): Softmax temperature.
        pad_id (int, optional): Padding token ID to ignore in CE loss.
        topk (int, optional): Number of top teacher tokens to keep.
                              If None or >= vocab_size, no truncation.
    """

    def __init__(
        self,
        alpha: float = 0.7,
        temperature: float = 5.0,
        pad_id: int = None,
        topk: int = None,
    ):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.topk = topk
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.ce_loss = (
            nn.CrossEntropyLoss(ignore_index=pad_id)
            if pad_id is not None
            else nn.CrossEntropyLoss()
        )

    def forward(self, student_logits, teacher_logits, labels, attention_mask=None):
        """
        Args:
            student_logits:  [B, seq_len, V]
            teacher_logits:  [B, seq_len, V]  (may be on a different device)
            labels:          [B, seq_len]      (unshifted input_ids)
            attention_mask:  [B, seq_len]      (1 = real token, 0 = pad)
        """
        # Ensure both logits are float32 and on the same device
        device = student_logits.device
        student_logits = student_logits.float()
        teacher_logits = teacher_logits.float().to(device)

        T = self.temperature
        B, S, V = teacher_logits.size()

        # ── Causal LM label shift ────────────────────────────────────────────
        # For next-token prediction the model predicts token[t+1] from token[t],
        # so we align: logits[:, :-1, :] predicts labels[:, 1:]
        shift_logits_s = student_logits[:, :-1, :].contiguous()   # [B, S-1, V]
        shift_logits_t = teacher_logits[:, :-1, :].contiguous()   # [B, S-1, V]
        shift_labels   = labels[:, 1:].contiguous()                # [B, S-1]

        # Build padding mask for the shifted positions
        if attention_mask is not None:
            # attention_mask[:, 1:] aligns with the shifted label positions
            pad_mask = attention_mask[:, 1:].bool()  # [B, S-1], True = real token
        else:
            pad_mask = (shift_labels != self.ce_loss.ignore_index
                        if hasattr(self.ce_loss, 'ignore_index')
                        else torch.ones(B, S - 1, dtype=torch.bool, device=device))

        BS1 = B * (S - 1)
        flat_s = shift_logits_s.view(BS1, V)
        flat_t = shift_logits_t.view(BS1, V)
        flat_mask = pad_mask.view(BS1)                 # [B*(S-1)]

        # ── Top-K truncation of teacher logits ──────────────────────────────
        if self.topk is not None and self.topk < V:
            topk_vals, topk_idx = torch.topk(flat_t, self.topk, dim=-1)
            inf_mask = torch.full_like(flat_t, float("-inf"))
            inf_mask.scatter_(1, topk_idx, topk_vals)
            flat_t_trunc = inf_mask
        else:
            flat_t_trunc = flat_t

        # ── KL divergence — only over non-padding positions ──────────────────
        # Selecting only real tokens prevents padding from inflating the loss.
        real_s = flat_s[flat_mask]          # [N_real, V]
        real_t = flat_t_trunc[flat_mask]    # [N_real, V]

        soft_teacher = torch.softmax(real_t / T, dim=-1)
        log_student  = torch.log_softmax(real_s / T, dim=-1)
        # KLDivLoss(batchmean) divides by batch size (N_real here)
        kl = self.kl_loss(log_student, soft_teacher) * (T * T)

        # ── Hard-label cross-entropy ──────────────────────────────────────────
        ce = self.ce_loss(flat_s, shift_labels.view(-1))

        return self.alpha * kl + (1.0 - self.alpha) * ce


# ─────────────────────────────────────────────
# 2. Data Preparation
# ─────────────────────────────────────────────

def load_and_preprocess_data(num_train_samples: int = 500):
    """
    Load the openassistant-guanaco dataset and format it as:
        ### Human: <question>
        ### Assistant: <answer>

    Args:
        num_train_samples: Number of training examples to use.

    Returns:
        small_dataset: Preprocessed training subset.
    """
    dataset = load_dataset("timdettmers/openassistant-guanaco", split="train")

    def preprocess_function(examples):
        prompts = []
        for txt in examples["text"]:
            # Dataset format uses "### Human:" / "### Assistant:" separators
            if "### Assistant:" in txt:
                human_part, assistant_part = txt.split("### Assistant:", 1)
            else:
                human_part = txt
                assistant_part = ""
            prompt = (
                human_part.strip()
                + "\n### Assistant: "
                + assistant_part.strip()
            )
            prompts.append(prompt)
        return {"text": prompts}

    # Select a subset to keep the experiment manageable
    small_dataset = (
        dataset
        .select(range(num_train_samples))
        .map(preprocess_function, batched=True)
    )
    return small_dataset


# ─────────────────────────────────────────────
# 3. Model Loading
# ─────────────────────────────────────────────

def load_models(teacher_name: str, student_name: str):
    """
    Load teacher and student causal LMs, plus the shared tokenizer.

    Uses device_map="auto" so HuggingFace automatically places layers across
    available GPUs/CPUs. Models are loaded in float32 for AMP training.

    Args:
        teacher_name: HuggingFace model ID for the teacher (e.g. "Qwen/Qwen3-4B").
        student_name: HuggingFace model ID for the student (e.g. "Qwen/Qwen3-0.6B").

    Returns:
        (tokenizer, teacher_model, student_model)
    """
    tokenizer = AutoTokenizer.from_pretrained(teacher_name)
    tokenizer.pad_token = tokenizer.eos_token  # Set padding token

    # Teacher — large model; no gradient required during training
    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_name, device_map="auto"
    )
    teacher_model.eval()

    # Student — smaller model; this is what we will train
    student_model = AutoModelForCausalLM.from_pretrained(
        student_name, device_map="auto"
    )

    return tokenizer, teacher_model, student_model


# ─────────────────────────────────────────────
# 4. Evaluation
# ─────────────────────────────────────────────

def evaluate_model(model, tokenizer, test_data) -> float:
    """
    Evaluate a model's perplexity on a test dataset.

    Perplexity (PPL) measures how well the model predicts the test tokens.
    Lower is better. After distillation the student's PPL should approach
    the teacher's PPL.

    Args:
        model: A HuggingFace causal LM.
        tokenizer: Matching tokenizer.
        test_data: A HuggingFace dataset with a "text" column.

    Returns:
        Perplexity score (float).
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for text in test_data["text"]:
            inputs = (
                tokenizer(text, return_tensors="pt", truncation=True)
                .to(model.device)
            )
            labels = inputs["input_ids"]
            outputs = model(**inputs, labels=labels)
            total_loss += outputs.loss.item()

    perplexity = torch.exp(
        torch.tensor(total_loss / len(test_data))
    ).item()
    return perplexity


# ─────────────────────────────────────────────
# 5. Training Loop
# ─────────────────────────────────────────────

def train_distillation(
    teacher_model,
    student_model,
    tokenizer,
    train_dataset,
    epochs: int = 3,
    batch_size: int = 2,
    lr: float = 5e-5,
    alpha: float = 0.7,
    temperature: float = 5.0,
    topk: int = 512,
):
    """
    Run the distillation fine-tuning loop.

    Key design decisions:
    - Teacher runs with torch.no_grad() to avoid tracking gradients.
    - AMP (Automatic Mixed Precision) reduces VRAM and speeds up training.
    - Gradient clipping (max norm 1.0) prevents gradient explosion.
    - Uses DistillationLossWithTopK for memory efficiency.

    Args:
        teacher_model: Frozen teacher LM.
        student_model: Student LM being trained.
        tokenizer:     Shared tokenizer.
        train_dataset: Preprocessed training dataset.
        epochs:        Number of training epochs.
        batch_size:    Samples per forward pass.
        lr:            AdamW learning rate.
        alpha:         Weight for KL distillation loss vs. CE loss.
        temperature:   Softmax temperature for soft labels.
        topk:          Top-K vocab entries to retain from teacher logits.
    """
    optimizer = optim.AdamW(student_model.parameters(), lr=lr, weight_decay=0.01)
    distill_loss_fn = DistillationLossWithTopK(
        alpha=alpha,
        temperature=temperature,
        pad_id=tokenizer.pad_token_id,
        topk=topk,
    )

    # Linear warmup scheduler: ramp LR from 0 → lr over the first 5% of steps,
    # then hold constant. Warmup prevents catastrophic forgetting at the start.
    total_steps   = (len(train_dataset) // batch_size) * epochs
    warmup_steps  = max(1, total_steps // 20)   # 5 % warmup
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Detect the student model's dtype to choose the right AMP strategy.
    #
    # float16  → narrow exponent range → needs GradScaler to prevent underflow
    # bfloat16 → same exponent range as float32 → no scaling needed
    #            (GradScaler raises NotImplementedError with bfloat16)
    param_dtype = next(student_model.parameters()).dtype
    use_scaler = (param_dtype == torch.float16)
    amp_dtype  = param_dtype
    scaler     = GradScaler("cuda") if use_scaler else None
    print(
        f"  AMP dtype : {amp_dtype}\n"
        f"  GradScaler: {'enabled' if use_scaler else 'disabled (bfloat16 — no loss scaling needed)'}\n"
    )

    for epoch in range(epochs):
        student_model.train()
        total_loss = 0.0
        num_batches = 0

        for i in range(0, len(train_dataset), batch_size):
            # ── Batch preparation ─────────────────────────────────────────
            batch_texts = train_dataset["text"][i : i + batch_size]
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            inputs = {k: v.to(student_model.device) for k, v in inputs.items()}
            labels = inputs["input_ids"].clone()

            optimizer.zero_grad()

            # ── Teacher forward (no gradient) ─────────────────────────────
            with torch.no_grad():
                teacher_outputs = teacher_model(
                    **inputs, output_hidden_states=False
                )
                # Cast to float32 for numerical stability in loss computation
                teacher_logits = teacher_outputs.logits.float()

            # ── Student forward + distillation loss (AMP) ─────────────────
            with autocast("cuda", dtype=amp_dtype):
                student_outputs = student_model(**inputs, labels=None)
                student_logits = student_outputs.logits
                loss = distill_loss_fn(
                    student_logits,
                    teacher_logits,
                    labels,
                    attention_mask=inputs.get("attention_mask"),
                )

            # ── Backward pass ─────────────────────────────────────────────
            if use_scaler:
                # float16: scale gradients to prevent underflow
                scaler.scale(loss).backward()
                # Unscale before clipping so clip_grad_norm sees real magnitudes
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                # bfloat16: standard backward, no scaling needed
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()
            total_loss += loss.item()
            num_batches += 1

            if num_batches % 50 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(
                    f"  [Epoch {epoch+1} | Step {num_batches:4d}] "
                    f"loss={loss.item():.4f}  lr={current_lr:.2e}"
                )

        avg_loss = total_loss / num_batches if num_batches else 0.0
        print(f"Epoch {epoch + 1}/{epochs}  |  Average Loss: {avg_loss:.4f}")

    return student_model


# ─────────────────────────────────────────────
# 6. Main
# ─────────────────────────────────────────────

def main():
    # ── Configuration ─────────────────────────────────────────────────────
    TEACHER_MODEL = "Qwen/Qwen3-4B"
    STUDENT_MODEL = "Qwen/Qwen3-0.6B"
    NUM_TRAIN_SAMPLES = 500
    NUM_TEST_SAMPLES  = 100
    EPOCHS     = 3
    BATCH_SIZE = 2
    LR         = 2e-5  # Conservative LR for pre-trained model (+ warmup scheduler)
    ALPHA      = 0.7   # Weight for KL distillation loss (vs. hard-label CE)
    TEMPERATURE = 5.0  # Softmax temperature (higher → softer distributions)
    TOPK       = 512   # Keep only top-512 teacher logits to save memory

    # ── Step 1: Data ──────────────────────────────────────────────────────
    print("Loading and preprocessing training data …")
    train_dataset = load_and_preprocess_data(NUM_TRAIN_SAMPLES)
    print(f"  → {len(train_dataset)} training samples ready.\n")

    # ── Step 2: Models ────────────────────────────────────────────────────
    print("Loading teacher and student models …")
    tokenizer, teacher_model, student_model = load_models(
        TEACHER_MODEL, STUDENT_MODEL
    )
    print(f"  → Teacher: {TEACHER_MODEL}")
    print(f"  → Student: {STUDENT_MODEL}\n")

    # ── Step 3: Baseline evaluation ───────────────────────────────────────
    print("Loading test data for evaluation …")
    test_dataset = load_dataset(
        "timdettmers/openassistant-guanaco", split="test"
    ).select(range(NUM_TEST_SAMPLES))

    print("Evaluating baseline perplexity (before distillation) …")
    teacher_ppl_before = evaluate_model(teacher_model, tokenizer, test_dataset)
    student_ppl_before = evaluate_model(student_model, tokenizer, test_dataset)
    print(f"  Teacher PPL (before): {teacher_ppl_before:.2f}")
    print(f"  Student PPL (before): {student_ppl_before:.2f}\n")

    # ── Step 4: Distillation training ─────────────────────────────────────
    print("Starting distillation training …")
    student_model = train_distillation(
        teacher_model=teacher_model,
        student_model=student_model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        alpha=ALPHA,
        temperature=TEMPERATURE,
        topk=TOPK,
    )
    print()

    # ── Step 5: Post-distillation evaluation ──────────────────────────────
    print("Evaluating perplexity after distillation …")
    student_ppl_after = evaluate_model(student_model, tokenizer, test_dataset)
    print(f"  Teacher PPL:          {teacher_ppl_before:.2f}")
    print(f"  Student PPL (before): {student_ppl_before:.2f}")
    print(f"  Student PPL (after):  {student_ppl_after:.2f}")

    improvement = student_ppl_before - student_ppl_after
    print(
        f"\n  Improvement: {improvement:+.2f} PPL "
        f"({'↓ better' if improvement > 0 else '↑ worse'})"
    )

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Summary & Considerations")
    print("=" * 60)
    print(
        """
Key factors that affect distillation quality:

1. Data Quality
   High-quality, diverse data improves distillation. Qwen3's pre-training
   data spans multiple languages and domains (code, math), which helps.

2. Hyperparameter Tuning
   - alpha:       controls the balance between KL loss and CE loss.
   - temperature: higher values soften the teacher distribution, exposing
                  more inter-token relationships ("dark knowledge").
   Too high a temperature may over-smooth; too low an alpha ignores the
   teacher signal.

3. Model Capacity Gap
   Qwen3-0.6B vs Qwen3-4B is a ~1:6.7 parameter ratio — a moderate gap
   that is generally manageable for distillation.
"""
    )


if __name__ == "__main__":
    main()
