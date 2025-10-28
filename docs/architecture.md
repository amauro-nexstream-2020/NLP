# System Architecture Documentation

## 1. Architecture Overview

The decoder-only transformer LLM follows the GPT-2 architecture with modern improvements. The system is designed to be modular, educational, and production-ready.

```
┌─────────────────────────────────────────────────────────────────┐
│                        Input Text                               │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      TOKENIZER                                  │
│  • BPE/WordPiece Algorithm                                      │
│  • Special Tokens: <PAD>, <UNK>, <BOS>, <EOS>                   │
│  • Vocabulary: 1K - 50K tokens                                  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼ Token IDs [batch, seq_len]
┌─────────────────────────────────────────────────────────────────┐
│                    EMBEDDING LAYER                              │
│  ┌──────────────────┐  ┌──────────────────┐                     │
│  │ Token Embeddings │  │ Positional       │                     │
│  │ [vocab, d_model] │+ │ Embeddings       │                     │
│  └──────────────────┘  │ [max_len,d_model]│                     │
│                        └──────────────────┘                     │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼ Embeddings [batch, seq_len, d_model]
┌─────────────────────────────────────────────────────────────────┐
│                  TRANSFORMER DECODER STACK                      │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │              Transformer Block × N                        │ │
│  │                                                           │ │
│  │  ┌─────────────────────────────────────────────────────┐ │ │
│  │  │  Layer Norm                                         │ │ │
│  │  └────────────┬────────────────────────────────────────┘ │ │
│  │               ▼                                           │ │
│  │  ┌─────────────────────────────────────────────────────┐ │ │
│  │  │  Multi-Head Self-Attention (Causal)                 │ │ │
│  │  │  • Query, Key, Value Projections                    │ │ │
│  │  │  • Scaled Dot-Product Attention                     │ │ │
│  │  │  • Causal Masking                                   │ │ │
│  │  │  • n_heads: 8-20                                    │ │ │
│  │  └────────────┬────────────────────────────────────────┘ │ │
│  │               ▼                                           │ │
│  │  ┌─────────────────────────────────────────────────────┐ │ │
│  │  │  Residual Connection                                │ │ │
│  │  └────────────┬────────────────────────────────────────┘ │ │
│  │               ▼                                           │ │
│  │  ┌─────────────────────────────────────────────────────┐ │ │
│  │  │  Layer Norm                                         │ │ │
│  │  └────────────┬────────────────────────────────────────┘ │ │
│  │               ▼                                           │ │
│  │  ┌─────────────────────────────────────────────────────┐ │ │
│  │  │  Feed-Forward Network                               │ │ │
│  │  │  • Linear: d_model → 4*d_model                      │ │ │
│  │  │  • GELU Activation                                  │ │ │
│  │  │  • Linear: 4*d_model → d_model                      │ │ │
│  │  └────────────┬────────────────────────────────────────┘ │ │
│  │               ▼                                           │ │
│  │  ┌─────────────────────────────────────────────────────┐ │ │
│  │  │  Residual Connection                                │ │ │
│  │  └─────────────────────────────────────────────────────┘ │ │
│  └───────────────────────────────────────────────────────────┘ │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FINAL LAYER NORM                             │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  OUTPUT PROJECTION (LM HEAD)                    │
│  • Linear: d_model → vocab_size                                │
│  • Weight Tying with Token Embeddings                          │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼ Logits [batch, seq_len, vocab_size]
┌─────────────────────────────────────────────────────────────────┐
│                      SOFTMAX                                    │
│  • Probability Distribution over Vocabulary                    │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
                   Generated Tokens
```

## 2. Component Details

### 2.1 Tokenizer
**Purpose**: Convert raw text to integer sequences

**Implementation**:
- Algorithm: Byte Pair Encoding (BPE)
- Library: Hugging Face `tokenizers`
- Vocabulary Size: Configurable (default: 50,257)

**Special Tokens**:
- `<PAD>`: Padding token (ID: 0)
- `<UNK>`: Unknown token (ID: 1)
- `<BOS>`: Beginning of sequence (ID: 2)
- `<EOS>`: End of sequence (ID: 3)

### 2.2 Embedding Layer
**Purpose**: Map token IDs to dense vectors

**Components**:
1. **Token Embeddings**: Learnable lookup table `[vocab_size, d_model]`
2. **Positional Embeddings**: Learnable position encodings `[max_seq_len, d_model]`

**Output**: Sum of token and positional embeddings

### 2.3 Multi-Head Self-Attention
**Purpose**: Allow model to attend to different positions

**Formula**:
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

**Implementation**:
- Query, Key, Value projections: `Linear(d_model, d_model)`
- Split into `n_heads` with dimension `d_k = d_model / n_heads`
- Apply scaled dot-product attention per head
- Concatenate heads and project: `Linear(d_model, d_model)`

**Causal Masking**: 
- Prevents attending to future positions
- Implemented via upper-triangular mask

### 2.4 Feed-Forward Network
**Purpose**: Position-wise transformation

**Architecture**:
```
FFN(x) = GELU(xW1 + b1)W2 + b2
```

**Dimensions**:
- Input: `d_model`
- Hidden: `4 * d_model` (typical expansion factor)
- Output: `d_model`

**Activation**: GELU (Gaussian Error Linear Unit)

### 2.5 Layer Normalization
**Purpose**: Stabilize training

**Formula**:
```
LayerNorm(x) = γ * (x - μ) / (σ + ε) + β
```

**Placement**: Pre-norm (GPT-2 style)
- Applied before attention and FFN
- More stable for deep models

### 2.6 Residual Connections
**Purpose**: Enable deep networks

**Formula**:
```
output = input + SubLayer(LayerNorm(input))
```

## 3. Data Flow

### 3.1 Forward Pass
```python
# Input: token IDs [batch, seq_len]
x = token_emb(input_ids) + pos_emb(positions)
x = dropout(x)

for layer in transformer_blocks:
    # Self-attention with residual
    x = x + attention(layer_norm(x))
    
    # FFN with residual
    x = x + ffn(layer_norm(x))

# Output projection
logits = lm_head(layer_norm(x))  # [batch, seq_len, vocab_size]
```

### 3.2 Training Loop
```python
for batch in dataloader:
    # Forward pass
    logits, loss = model(input_ids, targets)
    
    # Backward pass
    loss.backward()
    clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # Optimizer step
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
```

### 3.3 Generation (Autoregressive)
```python
for _ in range(max_new_tokens):
    # Forward pass (only on current sequence)
    logits = model(current_sequence)
    
    # Sample next token
    next_token = sample(logits[:, -1, :], temperature, top_k, top_p)
    
    # Append to sequence
    current_sequence = torch.cat([current_sequence, next_token], dim=1)
```

## 4. Model Configurations

### Tiny (10M parameters)
```python
{
    "n_embd": 128,
    "n_layer": 4,
    "n_head": 4,
    "n_positions": 512,
    "vocab_size": 10000
}
```

### Small (50M parameters)
```python
{
    "n_embd": 512,
    "n_layer": 8,
    "n_head": 8,
    "n_positions": 1024,
    "vocab_size": 50257
}
```

### Medium (350M parameters)
```python
{
    "n_embd": 1024,
    "n_layer": 24,
    "n_head": 16,
    "n_positions": 1024,
    "vocab_size": 50257
}
```

### Large (760M parameters)
```python
{
    "n_embd": 1280,
    "n_layer": 36,
    "n_head": 20,
    "n_positions": 1024,
    "vocab_size": 50257
}
```

## 5. Training Pipeline

### 5.1 Data Processing
```
Raw Text → Tokenization → Batching → Padding/Truncation → Model Input
```

### 5.2 Loss Function
```python
loss = CrossEntropyLoss(
    logits.view(-1, vocab_size),
    targets.view(-1),
    ignore_index=pad_token_id
)
```

### 5.3 Optimization
- **Optimizer**: AdamW with weight decay
- **Learning Rate**: 3e-4 (typical)
- **Schedule**: Warmup + Cosine decay
- **Gradient Clipping**: Max norm = 1.0

### 5.4 Mixed Precision Training
```python
with autocast():
    logits, loss = model(input_ids, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## 6. Inference Strategies

### 6.1 Greedy Decoding
Select token with highest probability at each step.

### 6.2 Top-K Sampling
Sample from top-k most probable tokens.

### 6.3 Top-P (Nucleus) Sampling
Sample from smallest set of tokens with cumulative probability ≥ p.

### 6.4 Temperature Scaling
```python
logits = logits / temperature  # temperature > 1 = more random
```

## 7. File Structure

```
NLP/
├── notebooks/          # Educational notebooks (01-09)
├── src/               # Core implementation
│   ├── model.py       # Transformer architecture
│   ├── tokenizer.py   # Tokenizer wrapper
│   ├── training.py    # Training utilities
│   ├── generation.py  # Text generation
│   └── utils.py       # Helper functions
├── config/            # Configuration files
│   ├── model_configs.py
│   └── training_configs.py
├── data/              # Datasets and tokenizers
├── checkpoints/       # Model checkpoints
├── results/           # Training logs
└── docs/              # Documentation
```

## 8. Extension Points

### 8.1 Custom Attention Mechanisms
- Flash Attention
- Linear Attention
- Sparse Attention

### 8.2 Parameter-Efficient Fine-Tuning
- LoRA (Low-Rank Adaptation)
- Adapters
- Prompt Tuning

### 8.3 Multimodal Extensions
- Vision encoder
- Audio encoder
- Cross-modal attention

## 9. Performance Considerations

### 9.1 Memory Optimization
- Gradient checkpointing
- Mixed precision (fp16/bf16)
- Batch size tuning

### 9.2 Speed Optimization
- Model compilation (`torch.compile`)
- Efficient kernels (FlashAttention)
- KV-cache for generation

### 9.3 Scaling
- Data parallel training
- Model parallel training (for > 1B params)
- Pipeline parallelism

---

**Version**: 1.0  
**Last Updated**: October 28, 2025
