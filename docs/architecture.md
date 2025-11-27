# Hybrid LLM Architecture (Mamba + Gated Attention)

This project now targets a 1.3B-parameter hybrid architecture that is trainable within ~14 days on a single A100/A6000 using streaming data. Everything from tokenization to training is scripted but not executed here.

## At a Glance
- **Backbone**: Mamba-2 SSM blocks interleaved with Gated Grouped-Query Attention (G-GQA) blocks.
- **Layout**: 24 layers, 18× Mamba + 6× Gated Attention (3:1 ratio), RMSNorm + SwiGLU MLP.
- **Hidden size**: 2048, **Heads**: 16 (4 KV heads), **Seq len (train)**: 4096.
- **Tokenizer**: Qwen2.5 BPE, vocab 50k (pad/eos aligned).
- **Parameters**: ~1.3B with untied LM head, embeddings initialized for stability.
- **Efficiency**: FlashAttention2, GQA, gated attention to suppress attention-sink, gradient checkpointing, 8-bit AdamW.

## Block Diagram
```
Tokens -> Embedding + RMSNorm
        -> [ Mamba | Mamba | Mamba | Gated Attention ] x 6
        -> RMSNorm -> LM Head (untied)
```
RoPE is applied inside attention blocks; Mamba-2 handles linear-time sequence modeling while attention layers provide exact retrieval checkpoints.

## Data & Streaming
- **Corpora**: 90% FineWeb-Edu (`HuggingFaceFW/fineweb-edu`, default `sample-100BT`) + 10% MedQA (`GBaker/MedQA-USMLE-4-options`).
- **Filtering**: Optional `language_score >= 0.99` for FineWeb-Edu quality.
- **Interleave**: `datasets.interleave_datasets` with probabilities; shuffle buffer 10k to decorrelate streaming shards.
- **Packing**: `hybrid_llm/data/streaming.py` chunks streams into `(seq_len+1)` tokens to produce `(input_ids, labels)` without staging to disk.

## Training Stack (ready-to-run)
- **Lightning module**: `hybrid_llm/training/lightning_module.py`
- **Data module**: `hybrid_llm/training/data_module.py`
- **Entry script**: `hybrid_llm/run_pretrain.py`
- **Configs**: `hybrid_llm/configs/model_config.py` and `hybrid_llm/configs/training_config.py`

### Default (single A100, 14-day target)
- Tokens: 100B (can be lowered further).
- Global batch: 512k tokens via micro-batch 8 × seq 4096 × grad_accum.
- Optimizer: AdamW (8-bit if available), betas (0.9, 0.95), wd 0.1, warmup 2k, cosine decay to 1e-5 scale.
- Precision: bf16-mixed; gradient clip 1.0.
- Logging: ClearML enabled by default; CSV fallback.
- Checkpoints: every 1k steps, keep last 5 in `checkpoints/`.

### Launch (not executed here)
```
python -m hybrid_llm.run_pretrain \
  --model-size large \
  --train-preset single_gpu
```
Use `--disable-clearml` to skip logging or `--max-steps` to truncate.

## File Map
- `hybrid_llm/model/hybrid_llm.py` — Hybrid architecture (Mamba-2 + gated attention + SwiGLU).
- `hybrid_llm/configs/model_config.py` — Model scales (nano→1.3B).
- `hybrid_llm/configs/training_config.py` — Training/data presets (14-day single GPU default).
- `hybrid_llm/data/streaming.py` — Streaming + interleaving FineWeb-Edu/MedQA, shuffle buffer, token packing.
- `hybrid_llm/training/lightning_module.py` — Lightning wrapper with optimizer/scheduler.
- `hybrid_llm/training/data_module.py` — Tokenizer + DataLoader wiring.
- `hybrid_llm/run_pretrain.py` — CLI launcher wired to ClearML checkpoints/logs.

## Why this design
- **Capacity vs. efficiency**: Mamba-2 gives linear-time throughput; sparse gated attention layers act as retrieval anchors.
- **Stability**: RMSNorm, gated attention, and zero-init output projections reduce early divergence (borrowed from nanochat/Qwen2.5 playbooks).
- **Storage-friendly**: Pure streaming; no Arrow conversion needed. Shuffle buffer mitigates locality in iterable datasets.
- **14-day feasibility**: 100B-token target with 40k tok/s assumption keeps wall-clock under the requested window; reduce tokens/model size further if needed.

## Next Steps
- Add evaluation stream for periodic perplexity on held-out FineWeb-Edu slices.
- Plug in PEFT/LoRA for downstream fine-tuning once pretraining converges.
- Optional vocab pruning pass (top-50k) if you want tighter embedding budget.
