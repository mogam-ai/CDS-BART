# 🧬 CDS-BART: A BART-Based Foundation Model for mRNA Sequence Analysis

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![uv](https://img.shields.io/badge/uv-package%20manager-orange)](https://github.com/astral-sh/uv)
[![Hydra](https://img.shields.io/badge/Config-Hydra-blue)](https://hydra.cc/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Hugging Face](https://img.shields.io/badge/🤗-Hugging%20Face-yellow)](https://huggingface.co/mogam-ai)

</div>

## 📋 Table of Contents
- [🎯 Overview](#-overview)
- [⚡ Quick Start](#-quick-start)
- [📦 Installation](#-installation)
- [🚀 Fine-tuning](#-fine-tuning)
- [⚙️ Configuration](#️-configuration)
- [📊 Datasets](#-datasets)
- [📚 Citations](#-citations)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

## 🎯 Overview

CDS-BART is designed as an easy-to-use tool, facilitating accessibility for researchers to leverage the development of mRNA vaccines and therapeutics. It is a BART-based foundation model that can be fine-tuned for various mRNA downstream tasks, such as:

- 🔬 **Protein expression prediction**
- 🧪 **mRNA stability prediction** 
- 🔄 **Riboswitch activity prediction**

The model is trained on a large dataset of mRNA sequences and freely available for use in the scientific community.

**🤗 Models & Datasets:** Available at [Hugging Face](https://huggingface.co/mogam-ai)

## ⚡ Quick Start

```bash
# Clone and setup
git clone https://github.com/mogam-ai/CDS-BART
cd CDS-BART
uv sync

# Run fine-tuning
uv run src/run_finetune.py +experiment=mRFP_expression.yaml \
  general.checkpoint_path="/your/checkpoint/path" \
  wandb_config.entity="your_wandb_entity"
```

## 📦 Installation

### 📋 Prerequisites
- Python 3.10+
- CUDA-compatible GPU (recommended)

### 🛠️ Setup Steps

1. **📥 Install uv package manager**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **📂 Clone the repository**
   ```bash
   git clone https://github.com/mogam-ai/CDS-BART
   cd CDS-BART
   ```

3. **📦 Install dependencies**
   ```bash
   uv sync
   ```

4. **🔧 Activate environment** (optional)
   ```bash
   source .venv/bin/activate
   ```
   > **Note:** Environment activation is not required when using `uv run` commands.

## 🚀 Fine-tuning

### 🎯 Available Experiments

Choose from six benchmark datasets:

| Dataset | Task Type | Description |
|---------|-----------|-------------|
| `ecoli_proteins.yaml` | Classification | E.coli protein expression levels |
| `fungal_expression.yaml` | Regression | Fungal gene expression |
| `mRFP_expression.yaml` | Regression | mRFP protein production |
| `mrna_stability.yaml` | Regression | mRNA stability across species |
| `sars_degrade.yaml` | Regression | SARS-CoV-2 vaccine degradation |
| `tc_riboswitch.yaml` | Regression | Riboswitch activity |

### 🔧 Required Configuration

Before running experiments, configure these parameters:

**1. 📁 Checkpoint Path**
```bash
uv run src/run_finetune.py +experiment=mRFP_expression.yaml \
  general.checkpoint_path="/your/local/checkpoint/directory"
```

**2. 📊 Weights & Biases (Optional)**
```bash
uv run src/run_finetune.py +experiment=mRFP_expression.yaml \
  wandb_config.entity="your_wandb_entity"
```

**3. 🔄 Combined Example**
```bash
uv run src/run_finetune.py +experiment=mRFP_expression.yaml \
  general.checkpoint_path="/home/user/checkpoints" \
  wandb_config.entity="your_entity"
```

## ⚙️ Configuration

### 📝 YAML File Modifications

Replace private values with these placeholders:

**Checkpoint Path:**
```yaml
general:
  checkpoint_path: "USER_MUST_SET_CHECKPOINT_PATH"  # Set your local checkpoint directory
```

**Wandb Entity:**
```yaml
wandb_config:
  entity: "USER_MUST_SET_WANDB_ENTITY"  # Set your wandb entity name if using wandb
```

## 📊 Datasets

Our benchmark datasets cover diverse mRNA analysis tasks:

### 🔬 **1. mRFP Expression**
- **Task:** Protein production prediction in E. coli
- **Target:** monomeric Red Fluorescent Protein expression levels
- **Reference:** Nieuwkoop, Thijs, et al. "Revealing determinants of translation efficiency via whole-gene codon randomization and machine learning." Nucleic acids research 51.5 (2023): 2363-2376.

### 🍄 **2. Fungal Expression**
- **Task:** Gene expression analysis in fungal species
- **Application:** Functional genomics research
- **Reference:** Wint R, Salamov A, Grigoriev IV. Kingdom-Wide Analysis of FungalProtein-Coding and tRNA Genes Reveals Conserved Patterns of Adaptive Evolution, Molecular Biology and Evolution (2022):39.

### 🦠 **3. E.coli Proteins**
- **Task:** Expression level classification (low/medium/high)
- **Data:** 2,308 low, 2,067 medium, 1,973 high expression samples
- **Reference:** Ding, Zundan, et al. "MPEPE, a predictive approach to improve protein expression in E. coli based on deep learning." Computational and Structural Biotechnology Journal 20 (2022): 1142-1153.

### 🧬 **4. mRNA Stability**
- **Task:** Cross-species mRNA stability prediction
- **Species:** Human, mouse, frog, fish
- **Reference:** Diez, Michay, et al. "iCodon customizes gene expression based on the codon composition." Scientific Reports 12.1 (2022): 12126.

### 🔄 **5. Tc-Riboswitch**
- **Task:** Riboswitch activity prediction
- **Target:** Tetracycline switching factor
- **Reference:** Groher, Ann-Christin, et al. "Tuning the performance of synthetic riboswitches using machine learning." ACS synthetic biology 8.1 (2018): 34-44

### 💉 **6. SARS-CoV-2 Vaccine**
- **Task:** mRNA vaccine degradation prediction
- **Target:** Degradation at 50°C with Mg²⁺ ions
- **Reference:** Leppek, Kathrin, et al. "Combinatorial optimization of mRNA structure, stability, and translation for RNA-based therapeutics." Nature communications 13.1 (2022): 1536.

## 📚 Citations

If you find CDS-BART useful in your research, please cite our paper:

```bibtex
@article{cds_bart_2025,
  title={CDS-BART: A BART-based model for codon-optimized protein sequence generation},
  author={Your Name},
  journal={under review},
  year={2025},
  volume={},
  number={},
  pages={},
  publisher={Oxford University Press}
}
```


## 🤝 Contributing

We welcome contributions!

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**🧬 Advancing mRNA Research with AI 🤖**

Made with ❤️ by the MOGAM Institute for Biomedical Research

</div>
