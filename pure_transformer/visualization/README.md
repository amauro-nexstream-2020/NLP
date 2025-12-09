# 🎨 Pure Transformer Interactive Visualizations

This directory contains interactive visualizations of the Pure Transformer architecture, including model components, training pipeline, and data flow.

## 📁 Files

### 1. **model_architecture.html** 
Interactive web-based visualization showing:
- All model components (Attention, MLP, Normalization, etc.)
- Methods and parameters for each component
- Architecture flow diagrams
- Code implementations
- Training pipeline visualization
- Searchable component browser

**How to use:**
```bash
# Open in browser
open pure_transformer/visualization/model_architecture.html

# Or use Python
python -c "import webbrowser; webbrowser.open('file://$(pwd)/pure_transformer/visualization/model_architecture.html')"
```

### 2. **model_exploration.ipynb**
Jupyter notebook with interactive analysis:
- Model instantiation and exploration
- Parameter distribution visualization
- Attention pattern analysis
- Model size comparisons
- Training metrics simulation
- Component dependency tree

**How to use:**
```bash
jupyter notebook pure_transformer/visualization/model_exploration.ipynb
```

### 3. **generate_viz.py**
Python script to generate visualizations programmatically:
- Analyze model structure
- Extract component information
- Generate visualization files
- Open browser automatically

**How to use:**
```bash
python pure_transformer/visualization/generate_viz.py
```

### 4. **model_analysis.json**
Auto-generated JSON file containing:
- Complete model structure
- Parameter counts per layer
- Layer types and configurations
- Total and trainable parameters

## 🎯 Key Features

### Interactive HTML Visualization
- **Component Browser**: Browse all model components organized by category
  - 🧠 Core Architecture (TransformerLM, Config)
  - 👁️ Attention Mechanisms (GQA, Sparse Attention)
  - ⚡ Layers & Blocks (RMSNorm, SwiGLU, Transformer Block)
  - 🎯 Training Components (GRPO, Pretrain)
  - 📊 Data Pipeline (Streaming, RL Prompts)

- **Detailed Views**: Click any component to see:
  - Description and purpose
  - Parameters and configurations
  - Available methods
  - Code implementation
  - Architecture flow diagram
  - Statistics and metrics

- **Search**: Quickly find components by name or type

- **Architecture Diagrams**: Visual flow charts showing:
  - Complete model forward pass
  - Transformer block structure
  - Attention mechanism flow
  - Training pipeline stages
  - RL fine-tuning process

### Jupyter Notebook Analysis
- **Parameter Analysis**: Visualize where model parameters are distributed
- **Configuration Comparison**: Compare TINY, SMALL, and MEDIUM configs
- **Training Simulation**: See learning rate schedules and loss curves
- **Interactive Exploration**: Modify and re-run analyses

## 🚀 Quick Start

1. **View the HTML visualization**:
   ```bash
   python pure_transformer/visualization/generate_viz.py
   ```
   This will automatically open the visualization in your browser.

2. **Explore in Jupyter**:
   ```bash
   jupyter notebook pure_transformer/visualization/model_exploration.ipynb
   ```

3. **Analyze your own model**:
   ```python
   from pure_transformer.visualization.generate_viz import analyze_model_structure
   from pure_transformer.model import TransformerLM
   from pure_transformer.configs import MEDIUM_CONFIG
   
   model = TransformerLM(MEDIUM_CONFIG)
   structure = analyze_model_structure(model)
   print(f"Total parameters: {structure['total_params']:,}")
   ```

## 📊 What You Can Visualize

### Model Components
- **TransformerLM**: Main model with 350M parameters
- **Attention**: Grouped Query Attention with RoPE
- **SwiGLU MLP**: Advanced feed-forward network
- **RMSNorm**: Efficient normalization layer
- **Sparse Attention**: DeepSeek-style lightning indexer
- **GRPO Trainer**: Enhanced RL training

### Training Pipeline
- Data loading and streaming
- Tokenization process
- Forward/backward passes
- Loss computation
- Optimizer steps
- RL fine-tuning stages

### Architecture Flow
- Token → Embeddings → Blocks → Output
- Pre-norm architecture with residuals
- Attention mechanism with RoPE
- SwiGLU activation in MLP

## 🎨 Customization

You can extend the visualizations by:

1. **Adding new components** to `model_architecture.html`:
   ```javascript
   components.your_category.push({
       name: 'YourComponent',
       type: 'Component Type',
       description: '...',
       params: {...},
       methods: ['method1', 'method2'],
       features: ['Feature 1', 'Feature 2'],
       code: `...`
   });
   ```

2. **Creating custom analysis notebooks**:
   - Copy `model_exploration.ipynb`
   - Add your own analysis cells
   - Use matplotlib/seaborn for custom plots

3. **Modifying the generator**:
   - Edit `generate_viz.py`
   - Add new analysis functions
   - Export to different formats

## 📈 Example Visualizations

The HTML visualization shows:
- **350M parameters** in MEDIUM config
- **24 transformer layers** with GQA
- **16 attention heads** (4 KV heads for efficiency)
- **2048 token context length**
- **3-day training** on single A100

The notebook includes:
- Bar charts of parameter distribution
- Learning rate schedules
- Training loss curves
- Model size comparisons
- Component dependency trees

## 🔧 Technical Details

### Technologies Used
- **D3.js**: For interactive web visualizations
- **Matplotlib/Seaborn**: For notebook plots
- **PyTorch**: For model introspection
- **JSON**: For data serialization

### Browser Requirements
- Modern browser with JavaScript enabled
- Recommended: Chrome, Firefox, Safari, Edge

### Python Requirements
- torch
- matplotlib (for notebooks)
- seaborn (for notebooks)
- jupyter (for notebooks)

## 📝 Notes

- The HTML visualization is completely self-contained (no external dependencies)
- All visualizations are generated from the actual model code
- The analysis is performed on the TINY config by default (for speed)
- You can analyze any config by modifying the generator script

## 🤝 Contributing

To add new visualizations:
1. Edit the component data in `model_architecture.html`
2. Add new analysis cells to `model_exploration.ipynb`
3. Extend `generate_viz.py` with new analysis functions
4. Update this README with your additions

## 📧 Support

For issues or questions about the visualizations:
- Check the code comments in each file
- Refer to the Pure Transformer main documentation
- Examine the generated `model_analysis.json` for model details
