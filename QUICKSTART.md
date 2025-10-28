# Quick Start Guide

Welcome to the Decoder-Only Transformer LLM course! This guide will help you get started quickly.

## 📦 Installation

### Step 1: Clone the Repository
```bash
git clone <your-repo-url>
cd NLP
```

### Step 2: Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import tokenizers; print('Tokenizers: OK')"
```

## 🎓 Learning Path

Follow the notebooks in sequential order:

### Week 1: Foundations
1. **`01_tokenizer.ipynb`** - Learn about tokenization and BPE
2. **`02_embeddings.ipynb`** - Token and positional embeddings
3. **`03_attention.ipynb`** - Self-attention mechanism

### Week 2: Architecture
4. **`04_transformer_block.ipynb`** - Build transformer blocks
5. **`05_model.ipynb`** - Assemble the full model
6. **`data_preparation.ipynb`** - Prepare datasets

### Week 3: Training & Generation
7. **`06_training.ipynb`** - Training pipeline
8. **`07_generation.ipynb`** - Text generation strategies
9. **`08_evaluation.ipynb`** - Model evaluation

### Week 4: Advanced Topics
10. **`09_fine_tuning.ipynb`** - Domain-specific fine-tuning

## 🚀 Quick Example

### Train a Tiny Model

```python
import torch
from src.model import DecoderLM
from src.tokenizer import BPETokenizerWrapper
from src.utils import set_seed
from config.model_configs import ModelConfigs

# Set seed for reproducibility
set_seed(42)

# Create tiny model configuration
config = ModelConfigs.tiny()

# Initialize model
model = DecoderLM(config)
print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")

# Train tokenizer
tokenizer = BPETokenizerWrapper(vocab_size=1000)
corpus = ["Hello world!", "How are you?", "This is a test."]
tokenizer.train(corpus)

# Encode some text
text = "Hello world"
tokens = tokenizer.encode(text)
print(f"Tokens: {tokens}")

# Decode back
decoded = tokenizer.decode(tokens)
print(f"Decoded: {decoded}")
```

### Generate Text

```python
from src.generation import generate_text

# After training...
prompt = "Once upon a time"
generated = generate_text(
    model=model,
    tokenizer=tokenizer,
    prompt=prompt,
    max_new_tokens=20,
    temperature=0.8,
    top_k=50
)
print(generated)
```

## 🎯 Project Milestones

### Milestone 1: Understanding (Week 1)
- [ ] Complete notebooks 1-3
- [ ] Understand tokenization, embeddings, attention
- [ ] Run example code successfully

### Milestone 2: Implementation (Week 2)
- [ ] Complete notebooks 4-6
- [ ] Build complete model architecture
- [ ] Prepare training dataset

### Milestone 3: Training (Week 3)
- [ ] Train small model on toy dataset
- [ ] Generate coherent text
- [ ] Evaluate model performance

### Milestone 4: Fine-Tuning (Week 4)
- [ ] Fine-tune on domain-specific data
- [ ] Compare with baseline
- [ ] Document results

## 💻 Training Environments

### Option 1: Local (CPU/GPU)
```bash
# Start Jupyter
jupyter notebook notebooks/
```

### Option 2: Google Colab
1. Upload notebooks to Google Drive
2. Open with Google Colab
3. Enable GPU: Runtime → Change runtime type → GPU

### Option 3: Nautilus HPC
```bash
# SSH to Nautilus cluster
ssh user@nautilus.cluster

# Load modules
module load python/3.9 cuda/11.8

# Run training
python -m src.training --config config/training_configs.py
```

## 📊 Expected Results

### Tiny Model (10M params)
- Training time: ~30 minutes on GPU
- Perplexity: ~50-100
- Use case: Learning and experimentation

### Small Model (50M params)
- Training time: ~2-4 hours on GPU
- Perplexity: ~30-50
- Use case: Prototyping and testing

### Medium Model (350M params)
- Training time: ~24-48 hours on GPU
- Perplexity: ~20-30
- Use case: Production experiments

## 🐛 Troubleshooting

### Issue: Out of Memory
**Solution**: Reduce batch size or model size
```python
config = ModelConfigs.tiny()  # Use smaller model
batch_size = 4  # Reduce batch size
```

### Issue: Slow Training
**Solution**: Enable mixed precision
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    loss = model(inputs)
```

### Issue: Poor Generation Quality
**Solution**: Train longer or adjust sampling
```python
# Increase training steps
max_steps = 10000

# Adjust temperature
generated = generate_text(..., temperature=0.7)  # Lower = more deterministic
```

## 📚 Additional Resources

### Documentation
- `docs/requirements.md` - Detailed requirements
- `docs/architecture.md` - Architecture diagrams
- `docs/testing.md` - Testing guidelines

### References
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [GPT-2 Paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [NanoGPT](https://github.com/karpathy/nanoGPT)

### Community
- GitHub Issues: Report bugs or ask questions
- Discussions: Share results and ideas

## 🎉 Next Steps

1. **Start with Notebook 1**: Open `notebooks/01_tokenizer.ipynb`
2. **Join the Discussion**: Share your progress
3. **Experiment**: Try different configurations
4. **Contribute**: Improve documentation or add features

Happy Learning! 🚀

---

**Questions?** Check the documentation or create an issue on GitHub.
