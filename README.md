# Decoder-Only Transformer LLM Course

A modular, educational implementation of a decoder-only transformer-based Large Language Model (LLM) for both general-purpose and domain-specific tasks.

## 🎯 Purpose

This project provides a hands-on, notebook-based course for understanding and implementing a transformer-based LLM from scratch. The system is designed for:
- Educational purposes (learning transformers step-by-step)
- Research contexts (domain-specific fine-tuning)
- Tasks: Q&A, summarization, dialogue, biomedical/chemical sequence analysis

**Target Model Size**: ≤1B parameters (training feasibility)

## 📚 Course Structure

The repository is organized as a series of Jupyter notebooks, each focusing on a specific component:

### Core Notebooks (Sequential Learning Path)

1. **`01_tokenizer.ipynb`** - Tokenization & Vocabulary
   - BPE/WordPiece implementation
   - Vocabulary building
   - Encoding/decoding text

2. **`02_embeddings.ipynb`** - Token & Positional Embeddings
   - Learned token embeddings
   - Positional encoding strategies
   - Embedding dimension analysis

3. **`03_attention.ipynb`** - Self-Attention Mechanism
   - Scaled dot-product attention
   - Multi-head attention
   - Causal masking for autoregressive generation

4. **`04_transformer_block.ipynb`** - Decoder Block
   - Layer normalization
   - Feed-forward networks
   - Residual connections
   - Complete transformer block assembly

5. **`05_model.ipynb`** - Full Model Architecture
   - Stacking decoder blocks
   - Output projection layer
   - Model initialization strategies

6. **`06_training.ipynb`** - Training Pipeline
   - Data loading & batching
   - Cross-entropy loss
   - Optimization (AdamW)
   - Learning rate scheduling
   - Checkpointing

7. **`07_generation.ipynb`** - Text Generation
   - Autoregressive sampling
   - Top-k and nucleus (top-p) sampling
   - Temperature control
   - Beam search (optional)

8. **`08_evaluation.ipynb`** - Model Evaluation
   - Perplexity calculation
   - Benchmark datasets
   - Validation metrics

9. **`09_fine_tuning.ipynb`** - Domain-Specific Fine-Tuning
   - Transfer learning strategies
   - Domain adaptation (biomedical, Q&A)
   - LoRA (parameter-efficient fine-tuning)

### Supplementary Notebooks

- **`data_preparation.ipynb`** - Dataset preprocessing
  - Textbook-quality datasets
  - Q&A dataset formatting
  - Domain-specific data (amino acids, medical)
  
- **`model_analysis.ipynb`** - Model inspection & visualization
  - Attention pattern visualization
  - Weight analysis
  - Embedding space exploration

- **`deployment.ipynb`** - Model deployment
  - Inference optimization
  - API creation
  - Simple web interface

## 🏗️ System Architecture

```
Input Text
    ↓
[Tokenizer] → Token IDs
    ↓
[Embedding Layer] → Token Embeddings + Positional Encoding
    ↓
[Decoder Stack]
    ├─ Multi-Head Self-Attention (Causal)
    ├─ Layer Normalization
    ├─ Feed-Forward Network
    └─ Residual Connections
    ↓
[Output Projection] → Logits
    ↓
[Softmax] → Probability Distribution
    ↓
Generated Text
```

## 📋 Requirements

### Functional Requirements
- **FR001**: Decoder-only transformer architecture
- **FR002**: Support for general-purpose and domain-specific datasets
- **FR003**: Custom/Hugging Face tokenizer integration
- **FR004**: Coherent Q&A response generation
- **FR005**: Modular training pipelines (Colab, JupyterHub, Nautilus)
- **FR006**: Cross-entropy loss optimization
- **FR007**: Fine-tuning capabilities for domain-specific applications
- **FR008**: Comprehensive documentation

### Non-Functional Requirements
- **NFR001**: Training within HPC/Colab session limits
- **NFR002**: Reproducible experiments via version control
- **NFR003**: Maintained documentation
- **NFR004**: Extensible for multimodal expansion
- **NFR005**: Baseline performance comparable to NanoGPT

## 🚀 Getting Started

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd NLP

# Install dependencies
pip install -r requirements.txt
```

### Quick Start

1. Start with `01_tokenizer.ipynb` to understand tokenization
2. Progress sequentially through notebooks 02-05 to build the model
3. Use `06_training.ipynb` to train on your dataset
4. Experiment with `07_generation.ipynb` for text generation
5. Fine-tune for specific domains using `09_fine_tuning.ipynb`

## 💻 Training Infrastructure

- **Nautilus**: NSF-funded NRP clusters (recommended for large-scale training)
- **Google Colab**: Free GPU access for experimentation
- **Local**: CPU/GPU training for small models

## 📊 Datasets

- **General-purpose**: Textbook-quality datasets, OpenWebText
- **Q&A**: SQuAD, Natural Questions
- **Domain-specific**: 
  - Biomedical: PubMed abstracts, protein sequences
  - Chemical: SMILES strings, molecular descriptions

## 🧪 Testing & Validation

### Test Suite
1. **Sanity Check**: Training on small Q&A dataset
2. **Generation Test**: Coherent responses on validation prompts
3. **Performance Test**: Baseline perplexity/accuracy on textbook datasets
4. **Domain Test**: Fine-tuned model performance on specialized data
5. **Documentation Review**: Completeness check

## 📁 Project Structure

```
NLP/
├── README.md
├── requirements.txt
├── config/
│   ├── model_configs.py      # Model architecture configurations
│   └── training_configs.py   # Training hyperparameters
├── notebooks/
│   ├── 01_tokenizer.ipynb
│   ├── 02_embeddings.ipynb
│   ├── 03_attention.ipynb
│   ├── 04_transformer_block.ipynb
│   ├── 05_model.ipynb
│   ├── 06_training.ipynb
│   ├── 07_generation.ipynb
│   ├── 08_evaluation.ipynb
│   ├── 09_fine_tuning.ipynb
│   ├── data_preparation.ipynb
│   ├── model_analysis.ipynb
│   └── deployment.ipynb
├── src/
│   ├── tokenizer.py          # Tokenizer utilities
│   ├── model.py              # Model architecture
│   ├── training.py           # Training utilities
│   ├── generation.py         # Text generation functions
│   └── utils.py              # Helper functions
├── data/
│   ├── raw/                  # Raw datasets
│   ├── processed/            # Preprocessed data
│   └── tokenizers/           # Saved tokenizer models
├── checkpoints/              # Model checkpoints
├── results/                  # Training logs, metrics
└── docs/
    ├── architecture.md       # Detailed architecture documentation
    ├── requirements.md       # Detailed requirements specification
    └── testing.md            # Test plans and results
```

## 🔮 Future Extensions

- [ ] Multimodal inputs (speech, images) - Qwen-style
- [ ] LoRA/parameter-efficient fine-tuning optimization
- [ ] Knowledge distillation from larger models
- [ ] Synthetic dataset generation
- [ ] Chatbot API deployment
- [ ] Web application interface

## 📖 References

- **NanoGPT**: [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
- **Attention is All You Need**: [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)
- **GPT-2**: [Radford et al., 2019](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

## 🤝 Contributing

This project is managed via GitHub with team collaboration:
- **Repository**: Managed by Anthony Mauro
- **Team**: Abhiram, Ananya, Anthony, Raunak



## 🙏 Acknowledgments

- NSF-funded Nautilus NRP clusters for computational resources
- Open-source community (PyTorch, Hugging Face)

---

**Acronyms**:
- **LLM**: Large Language Model
- **NLP**: Natural Language Processing
- **HPC**: High Performance Computing
- **GPU**: Graphics Processing Unit
- **Q&A**: Question and Answer
- **NRP**: National Research Platform
