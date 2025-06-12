
# CDS-BART: A BART-Based Foundation Model for mRNA Sequence Analysis

### CDS-BART
CDS-BART is designed as an easy-to-use tool, facilitating accessibility for researchers to leverage the development of mRNA vaccines and therapeutics. It is a BART-based foundation model that can be finetuned for various mRNA downstream tasks, such as protein expression prediction, mRNA stability prediction, and riboswitch activity prediction. The model is trained on a large dataset of mRNA sequences and freely available for use in the scientific community.
CDS-BART pretrained model and datasets are available at [Hugging Face](https://huggingface.co/mogam-ai/CDS-BART).

## Overview
This repository contains the code for training and evaluating the CDS-BART model, which is a BART-based model. The model is predict mRNA downstream of tasks. The model is trained on a large dataset of mRNA sequences.

<!-- Installation -->
## Installation

To install the necessary packages for this project, we use [`uv`](https://docs.astral.sh/uv/getting-started/installation/). This package management tool simplifies dependency management and ensures a reproducible environment.
<jp align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- Installation -->
### Steps to Install Packages

1. **Install uv**
   If you haven't installed `uv` yet, you can do so by following the instructions on the [uv installation page](https://docs.astral.sh/uv/getting-started/installation/). Run follwowing command in your terminal to install `uv`:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```


2. **Clone the Repository**
   First, clone the repository to your local machine using the following command:

   ```bash
   git clone https://github.com/mogam-ai/CDS-BART


3. **Navigate to the Project Directory**
   Change into the project directory:

   ```bash
   cd CDS-BART
   ```
   
4. **Install Dependencies**
   Use the `uv` package manager to install the required dependencies. Run the following command:

   ```bash
   uv sync
   ```

  This command will read the `project.toml` file in the project directory and install all specified packages. 

5. **Activate environment**
   ```bash
   source .venv/bin/activate
   ```
   you do not need to activate the environment if you are using uv run command. 

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- Finetune -->
## Finetune

```bash
python src/run_finetune.py +experiment=mRFP_expression.yaml

```
or using uv run command:
```bash 
uv run src/run_finetune.py  +experiment=mRFP_expression general.checkpoint_path="user_checkpoint_path"
```
### Explanation for Users:
In the command above, you need to replace user_checkpoint_path with your desired path where you want to save the finetuning checkpoints. There are six different YAML file options available for benchmark datasets, including:

- ecoli_proteins.yaml
- fungal_expression.yaml
- mRFP_expression.yaml
- mrna_stability.yaml
- sars.degrade.yaml
- tc_riboswitch.yaml

If you want to use a benchmark dataset, choose one of these YAML files. If you prefer to use your custom dataset, you will need to create a custom.yaml template similar to the existing ones, making only the necessary changes to reflect your custom dataset and any parameter adjustments.


<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- Pretrain -->
## Pretrain 

```bash

python src/run_pretrain.py +experiment=pretrain.yaml

```
<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- Finetune datasets -->
## Finetune datasets:

We used several datasets for finetuning the CDS-BART model. These datasets are crucial for training the model to understand and generate codon-optimized protein sequences. Each dataset is described below and can be accessed our huggingface repository [here](https://huggingface.co/mogam-ai).

1. mRFP expression data set(Nieuwkoop et al. 2023) : profiles protein production levels for several gene variants in Escherichia coli(E. coli). It measures the expression of monomeric Red Fluorescent Protein (mRFP), providing insights into how different gene variants influence protein production levels in bacterial systems.
2. Fungal expression data set (Grigoriev et al. 2014) :
provides a valuable resource for studying gene expression and functional genomics in fungal species.
3. E.coli protein data set (Ding et al. 2022) :
comprises experimental data for protein expression levels in E. coli, categorized as low, medium, or high expression. Specifically, it contains 2308 low expression, 2067 medium expression, and 1973 high expression data points, facilitating the study of protein production dynamics in bacterial cells.
4. mRNA stability (Diez et al. 2022) :
includes thousands of mRNA stability profiles obtained from various species, including humans, mice, frogs, and fish. It provides essential information on the stability of mRNA molecules across different organisms, contributing to our understanding of post-transcriptional regulation.
   includes thousands of mRNA stability profiles obtained from human, mouse, frog and fish.
5. Tc-riboswitch (Groher et al. 2019) :
consists of tetracycline (Tc) riboswitch dimer sequences positioned upstream of a Green Fluorescent Protein (GFP) mRNA. The measured variable is the switching factor, which quantifies the differential effect of the riboswitch in the presence or absence of tetracycline, providing insights into gene regulation mechanisms.
6. SARS-CoV-2 vaccine degradation (Leppek et al. 2022) :
encompasses a collection of mRNA sequences optimized for structural features, stability, and translation efficiency. The average degradation at 50°C with magnesium ions (deg_Mg_50C) values at each nucleotide position is used as the sequence-level target. This data is critical for understanding and improving the stability and efficacy of mRNA-based vaccines, such as those developed for SARS-CoV-2.
<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- Finetune datasets -->
## Dataset citations

1. Nieuwkoop, Thijs, et al. "Revealing determinants of translation efficiency via whole-gene codon randomization and machine learning." Nucleic acids research 51.5 (2023): 2363-2376.
2.
3. Ding, Zundan, et al. "MPEPE, a predictive approach to improve protein expression in E. coli based on deep learning." Computational and Structural Biotechnology Journal 20 (2022): 1142-1153.
4. Diez, Michay, et al. "iCodon customizes gene expression based on the codon composition." Scientific Reports 12.1 (2022): 12126.
5. Groher, Ann-Christin, et al. "Tuning the performance of synthetic riboswitches using machine learning." ACS synthetic biology 8.1 (2018): 34-44
6. Leppek, Kathrin, et al. "Combinatorial optimization of mRNA structure, stability, and translation for RNA-based therapeutics." Nature communications 13.1 (2022): 1536.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- Citation -->
## Citation
If you find the model useful in your research, please cite our paper:


```bibtex
@article{your_paper,
  title={CDS-BART: A BART-based model for codon-optimized protein sequence generation},
  author={Your Name},
  journal={Bioninformatics: Applicaiton Notes},
  year={2025},
  volume={},
  number={},
  pages={},
  publisher={.}
}
```
<p align="right">(<a href="#readme-top">back to top</a>)</p>