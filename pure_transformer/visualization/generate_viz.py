"""
Interactive Model Visualization Generator

Creates dynamic visualizations of the Pure Transformer architecture,
showing model components, data flow, and training pipeline.
"""

import json
import webbrowser
from pathlib import Path
from typing import Dict, List, Any
import torch
import torch.nn as nn


def analyze_model_structure(model: nn.Module) -> Dict[str, Any]:
    """
    Analyze a model's structure and extract component information.
    
    Args:
        model: PyTorch model to analyze
        
    Returns:
        Dictionary containing model structure information
    """
    structure = {
        "layers": [],
        "parameters": {},
        "total_params": 0,
        "trainable_params": 0,
    }
    
    # Count parameters
    for name, param in model.named_parameters():
        structure["parameters"][name] = {
            "shape": list(param.shape),
            "numel": param.numel(),
            "requires_grad": param.requires_grad,
            "dtype": str(param.dtype)
        }
        structure["total_params"] += param.numel()
        if param.requires_grad:
            structure["trainable_params"] += param.numel()
    
    # Analyze layers
    for name, module in model.named_modules():
        if name == "":
            continue
        
        layer_info = {
            "name": name,
            "type": module.__class__.__name__,
            "parameters": sum(p.numel() for p in module.parameters()),
        }
        
        # Add layer-specific info
        if hasattr(module, "in_features") and hasattr(module, "out_features"):
            layer_info["in_features"] = module.in_features
            layer_info["out_features"] = module.out_features
        
        if hasattr(module, "num_heads"):
            layer_info["num_heads"] = module.num_heads
        
        structure["layers"].append(layer_info)
    
    return structure


def create_component_graph_data() -> Dict[str, Any]:
    """Create graph data for component relationships."""
    
    nodes = [
        # Input/Output
        {"id": "input", "label": "Input Tokens", "type": "io", "color": "#4CAF50"},
        {"id": "output", "label": "Output Logits", "type": "io", "color": "#4CAF50"},
        
        # Embeddings
        {"id": "token_emb", "label": "Token Embeddings", "type": "embedding", "color": "#2196F3"},
        {"id": "rope", "label": "RoPE Cache", "type": "positional", "color": "#2196F3"},
        
        # Core blocks
        {"id": "transformer_blocks", "label": "24× Transformer Blocks", "type": "core", "color": "#9C27B0"},
        {"id": "attention", "label": "Multi-Head Attention", "type": "attention", "color": "#FF9800"},
        {"id": "mlp", "label": "SwiGLU MLP", "type": "ffn", "color": "#FF5722"},
        {"id": "norm1", "label": "RMSNorm", "type": "norm", "color": "#00BCD4"},
        {"id": "norm2", "label": "RMSNorm", "type": "norm", "color": "#00BCD4"},
        
        # Final layers
        {"id": "final_norm", "label": "Final RMSNorm", "type": "norm", "color": "#00BCD4"},
        {"id": "lm_head", "label": "LM Head", "type": "output", "color": "#E91E63"},
    ]
    
    edges = [
        {"source": "input", "target": "token_emb"},
        {"source": "token_emb", "target": "transformer_blocks"},
        {"source": "rope", "target": "attention"},
        {"source": "transformer_blocks", "target": "norm1"},
        {"source": "norm1", "target": "attention"},
        {"source": "attention", "target": "norm2"},
        {"source": "norm2", "target": "mlp"},
        {"source": "mlp", "target": "final_norm"},
        {"source": "final_norm", "target": "lm_head"},
        {"source": "lm_head", "target": "output"},
    ]
    
    return {"nodes": nodes, "edges": edges}


def generate_training_pipeline_viz() -> str:
    """Generate HTML visualization of the training pipeline."""
    
    return """
    <div class="pipeline-container">
        <h3>🚀 Training Pipeline</h3>
        <div class="pipeline-flow">
            <div class="pipeline-stage">
                <div class="stage-icon">📊</div>
                <div class="stage-title">Data Loading</div>
                <div class="stage-desc">Streaming datasets: FineWeb, FinePDFs, USMLE</div>
            </div>
            <div class="pipeline-arrow">→</div>
            
            <div class="pipeline-stage">
                <div class="stage-icon">🔤</div>
                <div class="stage-title">Tokenization</div>
                <div class="stage-desc">GPT-2 tokenizer, 2048 context</div>
            </div>
            <div class="pipeline-arrow">→</div>
            
            <div class="pipeline-stage">
                <div class="stage-icon">🧠</div>
                <div class="stage-title">Forward Pass</div>
                <div class="stage-desc">TransformerLM with Flash Attention</div>
            </div>
            <div class="pipeline-arrow">→</div>
            
            <div class="pipeline-stage">
                <div class="stage-icon">📉</div>
                <div class="stage-title">Loss Computation</div>
                <div class="stage-desc">Cross-entropy loss</div>
            </div>
            <div class="pipeline-arrow">→</div>
            
            <div class="pipeline-stage">
                <div class="stage-icon">⚡</div>
                <div class="stage-title">Backward Pass</div>
                <div class="stage-desc">Gradient computation</div>
            </div>
            <div class="pipeline-arrow">→</div>
            
            <div class="pipeline-stage">
                <div class="stage-icon">🎯</div>
                <div class="stage-title">Optimizer Step</div>
                <div class="stage-desc">AdamW with cosine schedule</div>
            </div>
        </div>
        
        <div style="margin-top: 30px;">
            <h4>RL Fine-tuning (GRPO)</h4>
            <div class="pipeline-flow">
                <div class="pipeline-stage">
                    <div class="stage-icon">❓</div>
                    <div class="stage-title">Prompt Sampling</div>
                    <div class="stage-desc">USMLE medical questions</div>
                </div>
                <div class="pipeline-arrow">→</div>
                
                <div class="pipeline-stage">
                    <div class="stage-icon">🤖</div>
                    <div class="stage-title">Generate Completions</div>
                    <div class="stage-desc">16 samples per prompt</div>
                </div>
                <div class="pipeline-arrow">→</div>
                
                <div class="pipeline-stage">
                    <div class="stage-icon">⭐</div>
                    <div class="stage-title">Reward Computation</div>
                    <div class="stage-desc">Compare with ground truth</div>
                </div>
                <div class="pipeline-arrow">→</div>
                
                <div class="pipeline-stage">
                    <div class="stage-icon">📊</div>
                    <div class="stage-title">Compute Advantages</div>
                    <div class="stage-desc">Group normalization</div>
                </div>
                <div class="pipeline-arrow">→</div>
                
                <div class="pipeline-stage">
                    <div class="stage-icon">🎓</div>
                    <div class="stage-title">GRPO Loss</div>
                    <div class="stage-desc">Policy gradient + KL penalty</div>
                </div>
            </div>
        </div>
    </div>
    """


def create_interactive_notebook():
    """Create a Jupyter notebook with interactive visualizations."""
    
    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 🚀 Pure Transformer - Interactive Model Exploration\n",
                    "\n",
                    "This notebook provides interactive visualizations and analysis of the Pure Transformer architecture."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "import torch\n",
                    "import torch.nn as nn\n",
                    "from pure_transformer.model import TransformerLM, TransformerConfig\n",
                    "from pure_transformer.configs import TINY_CONFIG, SMALL_CONFIG, MEDIUM_CONFIG\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "import numpy as np\n",
                    "\n",
                    "sns.set_style('whitegrid')\n",
                    "%matplotlib inline"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Model Instantiation\n",
                    "\n",
                    "Let's create a model and explore its structure."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Create a tiny model for exploration\n",
                    "model = TransformerLM(TINY_CONFIG)\n",
                    "print(f\"Model created with {model.count_parameters():,} parameters\")\n",
                    "print(f\"\\nModel configuration:\")\n",
                    "print(TINY_CONFIG)"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Parameter Distribution\n",
                    "\n",
                    "Visualize where parameters are distributed across the model."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Analyze parameter distribution\n",
                    "param_counts = {}\n",
                    "for name, module in model.named_modules():\n",
                    "    if len(list(module.children())) == 0:  # Leaf module\n",
                    "        params = sum(p.numel() for p in module.parameters())\n",
                    "        if params > 0:\n",
                    "            module_type = module.__class__.__name__\n",
                    "            param_counts[module_type] = param_counts.get(module_type, 0) + params\n",
                    "\n",
                    "# Plot\n",
                    "plt.figure(figsize=(12, 6))\n",
                    "colors = plt.cm.viridis(np.linspace(0, 1, len(param_counts)))\n",
                    "plt.bar(param_counts.keys(), param_counts.values(), color=colors)\n",
                    "plt.xticks(rotation=45, ha='right')\n",
                    "plt.ylabel('Number of Parameters')\n",
                    "plt.title('Parameter Distribution by Module Type')\n",
                    "plt.tight_layout()\n",
                    "plt.show()\n",
                    "\n",
                    "print(\"\\nParameter breakdown:\")\n",
                    "for module_type, count in sorted(param_counts.items(), key=lambda x: x[1], reverse=True):\n",
                    "    print(f\"{module_type:30s}: {count:12,} ({100*count/model.count_parameters():.1f}%)\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Attention Pattern Visualization\n",
                    "\n",
                    "Visualize attention patterns for a sample input."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Create sample input\n",
                    "sample_input = torch.randint(0, 1000, (1, 32))\n",
                    "\n",
                    "# Forward pass\n",
                    "with torch.no_grad():\n",
                    "    logits = model(sample_input)\n",
                    "\n",
                    "print(f\"Input shape: {sample_input.shape}\")\n",
                    "print(f\"Output shape: {logits.shape}\")\n",
                    "print(f\"\\nTop 5 predicted tokens: {logits[0, -1].topk(5).indices.tolist()}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. Model Size Comparison\n",
                    "\n",
                    "Compare different model configurations."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "configs = {\n",
                    "    'TINY': TINY_CONFIG,\n",
                    "    'SMALL': SMALL_CONFIG,\n",
                    "    'MEDIUM': MEDIUM_CONFIG,\n",
                    "}\n",
                    "\n",
                    "comparison = {}\n",
                    "for name, config in configs.items():\n",
                    "    temp_model = TransformerLM(config)\n",
                    "    comparison[name] = {\n",
                    "        'Parameters': temp_model.count_parameters(),\n",
                    "        'Layers': config.num_layers,\n",
                    "        'Hidden Size': config.hidden_size,\n",
                    "        'Heads': config.num_heads,\n",
                    "    }\n",
                    "\n",
                    "# Plot comparison\n",
                    "fig, axes = plt.subplots(2, 2, figsize=(14, 10))\n",
                    "fig.suptitle('Model Configuration Comparison', fontsize=16)\n",
                    "\n",
                    "for idx, (metric, ax) in enumerate(zip(['Parameters', 'Layers', 'Hidden Size', 'Heads'], axes.flat)):\n",
                    "    values = [comparison[name][metric] for name in configs.keys()]\n",
                    "    ax.bar(configs.keys(), values, color=['#4CAF50', '#2196F3', '#FF9800'])\n",
                    "    ax.set_title(metric)\n",
                    "    ax.set_ylabel('Value')\n",
                    "    if metric == 'Parameters':\n",
                    "        ax.set_ylabel('Parameters (Millions)')\n",
                    "        ax.set_yticklabels([f'{int(y/1e6)}M' for y in ax.get_yticks()])\n",
                    "\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 5. Training Metrics Simulation\n",
                    "\n",
                    "Simulate and visualize training metrics."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Simulate training curve\n",
                    "steps = np.arange(0, 10000, 100)\n",
                    "warmup_steps = 1000\n",
                    "max_lr = 3e-4\n",
                    "min_lr = 3e-5\n",
                    "\n",
                    "def cosine_schedule(step, warmup_steps, total_steps, max_lr, min_lr):\n",
                    "    if step < warmup_steps:\n",
                    "        return max_lr * step / warmup_steps\n",
                    "    progress = (step - warmup_steps) / (total_steps - warmup_steps)\n",
                    "    return min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(np.pi * progress))\n",
                    "\n",
                    "lrs = [cosine_schedule(s, warmup_steps, 10000, max_lr, min_lr) for s in steps]\n",
                    "\n",
                    "# Plot learning rate schedule\n",
                    "plt.figure(figsize=(12, 5))\n",
                    "plt.subplot(1, 2, 1)\n",
                    "plt.plot(steps, lrs, linewidth=2, color='#667eea')\n",
                    "plt.axvline(warmup_steps, color='red', linestyle='--', alpha=0.5, label='End of Warmup')\n",
                    "plt.xlabel('Training Step')\n",
                    "plt.ylabel('Learning Rate')\n",
                    "plt.title('Learning Rate Schedule (Cosine with Warmup)')\n",
                    "plt.legend()\n",
                    "plt.grid(alpha=0.3)\n",
                    "\n",
                    "# Simulate loss curve\n",
                    "plt.subplot(1, 2, 2)\n",
                    "loss = 8 - 5 * (1 - np.exp(-steps/2000)) + np.random.normal(0, 0.1, len(steps))\n",
                    "plt.plot(steps, loss, linewidth=2, color='#764ba2', alpha=0.7)\n",
                    "plt.xlabel('Training Step')\n",
                    "plt.ylabel('Loss')\n",
                    "plt.title('Training Loss Curve')\n",
                    "plt.grid(alpha=0.3)\n",
                    "\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 6. Component Interaction Map\n",
                    "\n",
                    "Visualize how different components interact."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "print(\"Component Dependencies:\")\n",
                    "print(\"\\nTransformerLM depends on:\")\n",
                    "print(\"  ├─ Token Embeddings\")\n",
                    "print(\"  ├─ RoPE Cache (precompute_rope_cache)\")\n",
                    "print(\"  ├─ Transformer Blocks (24×)\")\n",
                    "print(\"  │   ├─ RMSNorm\")\n",
                    "print(\"  │   ├─ Attention\")\n",
                    "print(\"  │   │   ├─ Q/K/V Projections\")\n",
                    "print(\"  │   │   ├─ apply_rotary_emb\")\n",
                    "print(\"  │   │   ├─ Flash Attention / SDPA\")\n",
                    "print(\"  │   │   └─ Output Projection\")\n",
                    "print(\"  │   ├─ RMSNorm\")\n",
                    "print(\"  │   └─ SwiGLUMLP\")\n",
                    "print(\"  │       ├─ Gate Projection\")\n",
                    "print(\"  │       ├─ Up Projection\")\n",
                    "print(\"  │       └─ Down Projection\")\n",
                    "print(\"  ├─ Final RMSNorm\")\n",
                    "print(\"  └─ LM Head\")"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    return notebook_content


def save_visualization_notebook(output_path: str = "model_exploration.ipynb"):
    """Save the interactive notebook."""
    notebook = create_interactive_notebook()
    
    output_file = Path(output_path)
    output_file.write_text(json.dumps(notebook, indent=2))
    print(f"✓ Saved interactive notebook to: {output_file.absolute()}")


def open_html_visualization():
    """Open the HTML visualization in the browser."""
    viz_path = Path(__file__).parent / "model_architecture.html"
    
    if viz_path.exists():
        webbrowser.open(f"file://{viz_path.absolute()}")
        print(f"✓ Opened visualization in browser: {viz_path}")
    else:
        print(f"✗ Visualization file not found: {viz_path}")


if __name__ == "__main__":
    import sys
    
    print("=" * 70)
    print("Pure Transformer - Interactive Visualization Generator")
    print("=" * 70)
    
    # Generate notebook
    print("\n1. Generating interactive Jupyter notebook...")
    save_visualization_notebook("pure_transformer/visualization/model_exploration.ipynb")
    
    # Open HTML visualization
    print("\n2. Opening HTML visualization...")
    open_html_visualization()
    
    # Analyze a model if available
    try:
        from pure_transformer.model import TransformerLM
        from pure_transformer.configs import TINY_CONFIG
        
        print("\n3. Analyzing TINY model...")
        model = TransformerLM(TINY_CONFIG)
        structure = analyze_model_structure(model)
        
        print(f"\nModel Analysis:")
        print(f"  Total parameters: {structure['total_params']:,}")
        print(f"  Trainable parameters: {structure['trainable_params']:,}")
        print(f"  Number of layers: {len(structure['layers'])}")
        
        # Save analysis
        analysis_path = Path("pure_transformer/visualization/model_analysis.json")
        analysis_path.parent.mkdir(parents=True, exist_ok=True)
        analysis_path.write_text(json.dumps(structure, indent=2))
        print(f"\n✓ Saved model analysis to: {analysis_path}")
        
    except ImportError as e:
        print(f"\n✗ Could not import model: {e}")
    
    print("\n" + "=" * 70)
    print("Visualization generation complete!")
    print("=" * 70)
