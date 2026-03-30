# 🇩🇪 German Small Language Model (Deutsch Geschichten)

A transformer-based small language model trained from scratch to generate German stories, exploring cross-lingual transfer learning and architectural experimentation.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📖 Overview

This project implements a GPT-2 style transformer architecture trained on German stories to investigate whether small language models can achieve comparable performance in non-English languages with limited data. The model is trained on a translated version of the TinyStories dataset, demonstrating cross-lingual capabilities.

### Key Features

- **Custom GPT-2 Architecture**: Built from scratch using PyTorch with multi-head attention, layer normalization, and residual connections
- **Cross-Lingual Dataset**: 200K English stories from Microsoft Research's TinyStories dataset translated to German using Meta's NLLB (No Language Left Behind) model
- **German Tokenization**: Utilizes `dbmdz/german-gpt2` tokenizer for proper German language processing
- **Modular Design**: Clean, extensible codebase for easy experimentation with different architectures and datasets

## 🏗️ Architecture

The model implements a decoder-only transformer architecture with the following specifications:

| Component | Configuration |
|-----------|--------------|
| Vocabulary Size | 50,266 tokens |
| Embedding Dimension | 384 |
| Context Length | 120 tokens |
| Number of Layers | 12 transformer blocks |
| Attention Heads | 2 heads per layer |
| Feed-Forward Scale | 4x embedding dimension |
| Dropout Rate | 0.1 |

### Model Components

```
SLM_Model
├── Embedding Layer (Token + Positional)
├── 12x Transformer Blocks
│   ├── Multi-Head Self-Attention (Causal)
│   ├── Layer Normalization
│   ├── Feed-Forward Network (GELU activation)
│   └── Residual Connections
├── Final Layer Normalization
└── Language Model Head
```

**Key Features:**
- **Causal Masking**: Prevents the model from attending to future tokens during training
- **Weight Tying**: Shares weights between embedding and output layers for parameter efficiency
- **Pre-Layer Normalization**: Applies normalization before attention and feed-forward blocks for training stability

## 📊 Dataset

### Source
- **Base Dataset**: [TinyStories](https://arxiv.org/abs/2305.07759) by Microsoft Research
- **Size**: 200,000 English children's stories
- **Translation**: Converted to German using Meta's NLLB-200 model

### Data Processing Pipeline

1. **Collection**: TinyStories dataset in English
2. **Translation**: Batch translation using NLLB model
3. **Tokenization**: German GPT-2 tokenizer (`dbmdz/german-gpt2`)
4. **Preprocessing**: 
   - Average token length: ~115 tokens
   - Maximum token length: 120 tokens
   - Format: JSONL with `{"de": "German story text"}` structure

### Sample Story Statistics
- **Total Stories**: 44,036 (processed)
- **Average Length**: 115.36 tokens
- **Max Length**: 120 tokens

## 🚀 Getting Started

### Prerequisites

```bash
python >= 3.10
torch >= 2.0
transformers >= 4.30.0
huggingface_hub
ipywidgets (for Jupyter notebooks)
```

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/german-slm.git
cd german-slm

# Install dependencies
pip install torch transformers huggingface_hub ipywidgets

# Authenticate with Hugging Face (required for tokenizer)
huggingface-cli login
```

### Quick Start

```python
from model import SLM_Model, Tokenizer

# Initialize model and tokenizer
model = SLM_Model()
tokenizer = Tokenizer("dbmdz/german-gpt2")

# Generate text
input_text = "Es war einmal"
tokens = tokenizer.encode([input_text])
output = model.forward(tokens)

# Decode output
generated_text = tokenizer.decode(output)
print(generated_text)
```

## 🔬 Research Goals

This project explores several research questions:

1. **Cross-Lingual Transfer**: Can models trained on translated data generate coherent stories in the target language?
2. **Data Efficiency**: How much translated data is needed for a small model to achieve meaningful performance?
3. **Architecture Scaling**: What is the optimal model size for low-resource story generation?
4. **Tokenization Impact**: How does tokenizer choice affect model performance on morphologically rich languages?

## 🛠️ Planned Experiments

- [ ] **Custom Tokenizer**: Train a BPE tokenizer specifically on German story data
- [ ] **Extended Dataset**: Generate additional synthetic stories using GPT-4
- [ ] **Architecture Variants**: 
  - Experiment with different layer counts (6, 12, 24)
  - Test various attention head configurations
  - Compare different context lengths
- [ ] **Data Augmentation**: Apply back-translation and paraphrasing
- [ ] **Evaluation Metrics**: Implement perplexity, BLEU, and human evaluation
- [ ] **Fine-tuning**: Adapt pre-trained German models for comparison

## 📁 Project Structure

```
german-slm/
├── model.py              # Core model implementation
├── training.py           # Training pipeline and data loading
├── tokenizer.py          # Tokenizer wrapper
├── dataset/
│   └── dataset.jsonl     # Translated German stories
├── notebooks/
│   └── SLM.ipynb        # Development notebook
├── checkpoints/          # Model weights
└── README.md
```

## 🎯 Current Status

- ✅ GPT-2 architecture implementation
- ✅ Dataset translation (200K stories)
- ✅ German tokenizer integration
- ✅ Data preprocessing pipeline
- 🚧 Training loop implementation
- 🚧 Evaluation metrics
- 📋 Model optimization
- 📋 Custom tokenizer training

## 📈 Performance Metrics

*Coming soon - metrics will be added after training completion*

## 🤝 Contributing

Contributions are welcome! Areas of interest:

- Training optimization techniques
- Alternative architectures
- Dataset expansion
- Evaluation methodology
- Documentation improvements

## 📚 References

- [TinyStories: How Small Can Language Models Be and Still Speak Coherent English?](https://arxiv.org/abs/2305.07759)
- [No Language Left Behind (NLLB)](https://ai.meta.com/research/no-language-left-behind/)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Language Models are Unsupervised Multitask Learners (GPT-2)](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Microsoft Research for the TinyStories dataset
- Meta AI for the NLLB translation model
- Hugging Face for tokenizer infrastructure
- The open-source ML community

## 📧 Contact

For questions or collaboration opportunities, please open an issue or reach out via [your contact method].

---

**Note**: This is an educational research project exploring small language model capabilities in German. The model is not intended for production use.
