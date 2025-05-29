
# CDS-BART

Repository for the paper "CDS-BART: A BART-based model for codon-optimized protein sequence generation".

## Overview
This repository contains the code for training and evaluating the CDS-BART model, which is a BART-based model. The model is predict mRNA downstream of tasks. The model is trained on a large dataset of mRNA sequences.
### Installation
 
Dependency management is using conda. To create the environment, run:
```
conda env create -f environment.yaml

```

## Pretrain 

```bash

python src/run_pretrain.py +experiment=pretrain.yaml

```

## Finetune

```bash
python src/run_finetune.py +experiment=finetune_codon_bert.yaml

```

Finetune data:
1. mRFP expression data set(Nieuwkoop et al. 2023) profiles protein production levels for several gene variants in E.coli.
2. 

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
