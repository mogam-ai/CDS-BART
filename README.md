
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
1. mRFP expression data set(Nieuwkoop et al. 2023) :
   profiles protein production levels for several gene variants in E.coli.
2. Fungal expression data set (Grigoriev et al. 2014) :
   includes CDSs >150 bp from a wide range of fungal ge- nomes
3. E.coli protein data set (Ding et al. 2022) :
   comprises experimental data for protein expression in E.coli, which are labeled as low, medium, or high expression (2308, 2067, and 1973)
4. mRNA stability (Diez et al. 2022) :
   includes thousands of mRNA stability profiles obtained from human, mouse, frog and fish.
5. Tc-riboswitch (Groher et al. 2019) :
   consists of a set of tetracycline (Tc) riboswitch dimer sequences upstream of a GFP mRNA. The measured variable in this data set is the switching factor, which refers to the differential effect of the riboswitch in the presence or abscence of Tc.
6. SARS-CoV-2 vaccine degradation (Leppek et al. 2022) :
   encompasses a set of mRNA sequences that have been tuned for their structural features, stability, and translation efficiency. The average of the deg_Mg_50C values at each nucleotide is treated as the sequence-level target.

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
