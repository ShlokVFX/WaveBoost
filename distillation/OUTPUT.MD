➜  distillation git:(master) ✗ uv run qwen3_distillation.py                               
Loading and preprocessing training data …
Repo card metadata block was not found. Setting CardData to empty.
  → 500 training samples ready.

Loading teacher and student models …
Loading weights: 100%|█| 398/398 [00:08<00:00, 46.62it/s, Materializing param=model.norm.w
  → 500 training samples ready.

Loading teacher and student models …
Loading weights: 100%|█| 398/398 [00:08<00:00, 46.62it/s, Materializing param=model.norm.w
Loading weights: 100%|█| 311/311 [00:01<00:00, 189.59it/s, Materializing param=model.norm.
The tied weights mapping and config for this model specifies to tie model.embed_tokens.weight to lm_head.weight, but both are present in the checkpoints, so we will NOT tie them. You should update the config with `tie_word_embeddings=False` to silence this warning
  → Teacher: Qwen/Qwen3-4B
  → Student: Qwen/Qwen3-0.6B

Loading test data for evaluation …
Repo card metadata block was not found. Setting CardData to empty.
Evaluating baseline perplexity (before distillation) …
  Teacher PPL (before): 6.99
  Student PPL (before): 10.06

Starting distillation training …
  AMP dtype : torch.bfloat16
  GradScaler: disabled (bfloat16 — no loss scaling needed)     

Epoch 1/3  |  Average Loss: 18636.9138
Epoch 2/3  |  Average Loss: 15850.6440
Epoch 3/3  |  Average Loss: 14952.9684

============================================================
Summary & Considerations
============================================================

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

➜  distillation git:(master) ✗